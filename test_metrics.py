import pytest
import json
import httpx
import database as db
from core.router import RouterService
from core.rate_limiter import RateLimiter
from core.providers.factory import ProviderFactory

@pytest.mark.anyio
async def test_metrics_aggregation(tmp_path):
    import os
    db_file = os.path.join(tmp_path, "test_metrics.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    # Add some mock logs
    db.add_log("OpenAI", "POST", "/v1/chat", "{}", 200, "{}", tokens_sent=10, tokens_received=20, latency_ms=100)
    db.add_log("OpenAI", "POST", "/v1/chat", "{}", 200, "{}", tokens_sent=5, tokens_received=15, latency_ms=200)
    db.add_log("Anthropic", "POST", "/v1/messages", "{}", 200, "{}", tokens_sent=50, tokens_received=100, latency_ms=500)

    summary = db.get_metrics_summary()
    assert len(summary) == 2

    openai_metrics = next(m for m in summary if m["provider_name"] == "OpenAI")
    assert openai_metrics["request_count"] == 2
    assert openai_metrics["total_tokens_sent"] == 15
    assert openai_metrics["total_tokens_received"] == 35
    assert openai_metrics["avg_latency"] == 150.0

    anthropic_metrics = next(m for m in summary if m["provider_name"] == "Anthropic")
    assert anthropic_metrics["request_count"] == 1
    assert anthropic_metrics["total_tokens_sent"] == 50
    assert anthropic_metrics["total_tokens_received"] == 100
    assert anthropic_metrics["avg_latency"] == 500.0

@pytest.mark.anyio
async def test_router_metrics_capture_non_streaming(tmp_path, respx_mock):
    import os
    db_file = os.path.join(tmp_path, "test_router_metrics.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    # Mock provider
    db.add_provider("TestProvider", "openai", "https://api.openai.com/v1/chat/completions", "sk-test", "gpt-4", is_active=1)
    
    # Mock OpenAI response
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 123456789,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    ))

    async with httpx.AsyncClient() as client:
        router = RouterService(http_client=client, rate_limiter=RateLimiter(100))
        # Invalidate cache to ensure it reads from the new DB
        router.provider_service.reload_active_provider()
        
        # Make a request
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        await router.route_openai_request(request_body, stream=False)

    # Check logs
    logs = db.get_logs()
    assert len(logs) == 1
    log = logs[0]
    assert log["provider_name"] == "TestProvider"
    assert log["tokens_sent"] == 10
    assert log["tokens_received"] == 20
    assert log["latency_ms"] >= 0

@pytest.mark.anyio
async def test_router_metrics_capture_streaming(tmp_path, respx_mock):
    import os
    db_file = os.path.join(tmp_path, "test_router_metrics_stream.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    # Mock provider
    db.add_provider("StreamProvider", "openai", "https://api.openai.com/v1/chat/completions", "sk-test", "gpt-4", is_active=1)
    
    # Mock OpenAI streaming response
    stream_content = (
        'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": null}]}\n\n'
        'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": null}]}\n\n'
        'data: {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}], "usage": {"prompt_tokens": 15, "completion_tokens": 5}}\n\n'
        'data: [DONE]\n\n'
    )
    
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=httpx.Response(
        200,
        content=stream_content,
        headers={"Content-Type": "text/event-stream"}
    ))

    async with httpx.AsyncClient() as client:
        router = RouterService(http_client=client, rate_limiter=RateLimiter(100))
        # Invalidate cache to ensure it reads from the new DB
        router.provider_service.reload_active_provider()
        
        # Make a request
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }
        response = await router.route_openai_request(request_body, stream=True)
        
        # Consume the stream
        async for _ in response.body_iterator:
            pass

    # Check logs
    logs = db.get_logs()
    assert len(logs) == 1
    log = logs[0]
    assert log["provider_name"] == "StreamProvider"
    assert log["tokens_sent"] == 15
    assert log["tokens_received"] == 5
    assert log["latency_ms"] >= 0

@pytest.mark.anyio
async def test_metrics_api_endpoint(tmp_path):
    import os
    from fastapi.testclient import TestClient
    from main import app
    
    db_file = os.path.join(tmp_path, "test_api_metrics.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    # Add mock logs
    db.add_log("ProviderA", "POST", "/v1/chat", "{}", 200, "{}", tokens_sent=10, tokens_received=20, latency_ms=100)
    
    client = TestClient(app)
    response = client.get("/api/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["provider_name"] == "ProviderA"
    assert data[0]["request_count"] == 1
    assert data[0]["total_tokens_sent"] == 10
    assert data[0]["avg_latency"] == 100.0

@pytest.mark.anyio
async def test_router_metrics_estimation_fallback(tmp_path, respx_mock):
    import os
    db_file = os.path.join(tmp_path, "test_router_metrics_est.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    # Mock provider
    db.add_provider("EstProvider", "openai", "https://api.openai.com/v1/chat/completions", "sk-test", "gpt-4", is_active=1)
    
    # Mock response WITHOUT usage data
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello world! This is a test."}
            }]
        }
    ))

    async with httpx.AsyncClient() as client:
        router = RouterService(http_client=client, rate_limiter=RateLimiter(100))
        router.provider_service.reload_active_provider()
        
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "How many tokens is this?"}]
        }
        await router.route_openai_request(request_body, stream=False)

    # Check logs
    logs = db.get_logs()
    assert len(logs) == 1
    log = logs[0]
    
    # Estimation: "How many tokens is this?" = 25 chars -> ~6 tokens
    # Estimation: "Hello world! This is a test." = 28 chars -> ~7 tokens
    assert log["tokens_sent"] > 0
    assert log["tokens_received"] > 0
    assert log["latency_ms"] >= 0
    
    # Verify exact values based on 4 chars per token heuristic
    # "How many tokens is this?" -> 25 chars // 4 = 6
    # "Hello world! This is a test." -> 28 chars // 4 = 7
    assert log["tokens_sent"] == 6
    assert log["tokens_received"] == 7
