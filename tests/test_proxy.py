import pytest
import json
import time
import database as db
import httpx
from core.providers.translation import (
    anthropic_to_openai_request,
    openai_to_anthropic_request,
    sanitize_openai_payload,
    openai_to_anthropic_response,
    anthropic_to_openai_response,
    SIGNATURE_SEPARATOR
)
from core.providers.factory import ProviderFactory
from core.providers.base import BaseProvider
from core.providers.openai import OpenAIProvider
from core.providers.anthropic import AnthropicProvider
from core.providers.gemini import GeminiProvider
from core.providers.mistral import MistralProvider
from core.providers.openrouter import OpenRouterProvider

# --- Unit Tests for Providers ---

def test_provider_factory_instantiation():
    factory = ProviderFactory()
    
    # Test OpenAI instantiation
    openai_config = {
        "name": "Test OpenAI",
        "api_type": "openai",
        "endpoint_url": "https://api.openai.com/v1/chat/completions",
        "api_key": "sk-test",
        "model_name": "gpt-4o",
        "is_active": 1,
        "id": 1
    }
    provider = factory.create_provider(openai_config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "Test OpenAI"
    assert provider.api_type == "openai"
    
    # Test Anthropic instantiation
    anthropic_config = {
        "name": "Test Anthropic",
        "api_type": "anthropic",
        "endpoint_url": "https://api.anthropic.com/v1/messages",
        "api_key": "sk-ant-test",
        "model_name": "claude-3-5-sonnet",
        "is_active": 0,
        "id": 2
    }
    provider = factory.create_provider(anthropic_config)
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "Test Anthropic"
    assert provider.api_type == "anthropic"
    
    # Test Gemini instantiation
    gemini_config = {
        "name": "Test Gemini",
        "api_type": "gemini",
        "endpoint_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key": "ai-test",
        "model_name": "gemini-1.5-pro",
        "is_active": 0,
        "id": 3
    }
    provider = factory.create_provider(gemini_config)
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "Test Gemini"
    assert provider.api_type == "gemini"
    
    # Test Mistral instantiation
    mistral_config = {
        "name": "Test Mistral",
        "api_type": "mistral",
        "endpoint_url": "https://api.mistral.ai/v1/chat/completions",
        "api_key": "mistral-test",
        "model_name": "mistral-large-latest",
        "is_active": 0,
        "id": 4
    }
    provider = factory.create_provider(mistral_config)
    assert isinstance(provider, MistralProvider)
    assert provider.name == "Test Mistral"
    assert provider.api_type == "mistral"
    
    # Test OpenRouter instantiation
    openrouter_config = {
        "name": "Test OpenRouter",
        "api_type": "openrouter",
        "endpoint_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": "sk-or-test",
        "model_name": "meta-llama/llama-3-70b",
        "is_active": 0,
        "id": 5
    }
    provider = factory.create_provider(openrouter_config)
    assert isinstance(provider, OpenRouterProvider)
    assert provider.name == "Test OpenRouter"
    assert provider.api_type == "openrouter"

# --- Unit Tests for Translator ---

def test_anthropic_to_openai_request_translation():
    # Simple message
    anth_req = {
        "model": "claude-3-5-sonnet",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True
    }
    
    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")
    
    assert openai_req["model"] == "gpt-4o"
    assert len(openai_req["messages"]) == 2
    assert openai_req["messages"][0]["role"] == "system"
    assert openai_req["messages"][0]["content"] == "You are a helpful assistant."
    assert openai_req["messages"][1]["role"] == "user"
    assert openai_req["messages"][1]["content"] == "Hello!"
    assert openai_req["max_tokens"] == 1024
    assert openai_req["temperature"] == 0.7
    assert openai_req["stream"] is True

def test_anthropic_to_openai_request_with_tools():
    # Message with tool use definitions and tool results
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Command output text here"
                    }
                ]
            }
        ],
        "tools": [
            {
                "name": "run_command",
                "description": "Runs a command in shell",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
    
    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")
    
    assert len(openai_req["tools"]) == 1
    assert openai_req["tools"][0]["type"] == "function"
    assert openai_req["tools"][0]["function"]["name"] == "run_command"
    assert openai_req["tools"][0]["function"]["parameters"]["properties"]["command"]["type"] == "string"
    
    # Check messages array has the role tool
    assert len(openai_req["messages"]) == 1
    assert openai_req["messages"][0]["role"] == "tool"
    assert openai_req["messages"][0]["tool_call_id"] == "toolu_123"
    assert openai_req["messages"][0]["content"] == "Command output text here"

def test_openai_to_anthropic_response_translation():
    # OpenAI response with tool calls
    openai_res = {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling command now...",
                    "tool_calls": [
                        {
                            "id": "call_999",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": "{\"command\": \"ls\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25
        }
    }
    
    anth_res = openai_to_anthropic_response(openai_res)
    
    assert anth_res["role"] == "assistant"
    assert anth_res["stop_reason"] == "tool_use"
    assert len(anth_res["content"]) == 2
    
    assert anth_res["content"][0]["type"] == "text"
    assert anth_res["content"][0]["text"] == "Calling command now..."
    
    assert anth_res["content"][1]["type"] == "tool_use"
    assert anth_res["content"][1]["id"] == "call_999"
    assert anth_res["content"][1]["name"] == "run_command"
    assert anth_res["content"][1]["input"] == {"command": "ls"}
    
    assert anth_res["usage"]["input_tokens"] == 15
    assert anth_res["usage"]["output_tokens"] == 25

def test_gemini_thought_signature_translation():
    # 1. Test translation from Gemini response (OpenAI format) to Anthropic response
    # This simulates receiving a tool call from Gemini with a signature
    gemini_res = {
        "id": "chatcmpl-gemini",
        "model": "gemini-2.0-flash-thinking",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Thinking...",
                    "tool_calls": [
                        {
                            "id": "call_gemini_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"London\"}"
                            },
                            "extra_content": {
                                "google": {
                                    "thought_signature": "SIG_DATA_ABC_123"
                                }
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }
    
    anth_res = openai_to_anthropic_response(gemini_res)
    
    # The ID should now be "mangled" with the signature
    expected_id = f"call_gemini_123{SIGNATURE_SEPARATOR}SIG_DATA_ABC_123"
    assert anth_res["content"][1]["id"] == expected_id
    
    # 2. Test translation from Anthropic request (with mangled ID) back to Gemini (OpenAI format)
    # This simulates the next turn where the client sends back history or a tool result
    anth_req = {
        "model": "gemini-2.0-flash-thinking",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking..."},
                    {
                        "type": "tool_use",
                        "id": expected_id,
                        "name": "get_weather",
                        "input": {"location": "London"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": expected_id,
                        "content": "Sunny, 20C"
                    }
                ]
            }
        ]
    }
    
    openai_req = anthropic_to_openai_request(anth_req, "gemini-2.0-flash-thinking")
    
    # Check assistant message has the restored signature in extra_content
    assistant_msg = openai_req["messages"][0]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_gemini_123"
    assert assistant_msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG_DATA_ABC_123"
    
    # Check tool result message has the cleaned ID (no signature)
    tool_msg = openai_req["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_gemini_123"
    assert tool_msg["content"] == "Sunny, 20C"

def test_mistral_sanitization():
    # Simulate an Anthropic request with cache_control and list-based system prompt
    anth_req = {
        "model": "mistral-large-latest",
        "system": [{"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"call_123{SIGNATURE_SEPARATOR}SIG_STUFF",
                        "name": "tool",
                        "input": {}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
                ]
            }
        ]
    }
    
    # 1. Translate to OpenAI
    openai_req = anthropic_to_openai_request(anth_req, "mistral-large-latest")
    
    # 2. Sanitize for Mistral (is_gemini=False)
    sanitized_mistral = sanitize_openai_payload(openai_req, is_gemini=False)
    
    # System prompt should be a string and clean
    assert sanitized_mistral["messages"][0]["role"] == "system"
    assert sanitized_mistral["messages"][0]["content"] == "System prompt"
    
    # Assistant message should have extra_content STRIPPED
    assert sanitized_mistral["messages"][1]["role"] == "assistant"
    assert "extra_content" not in sanitized_mistral["messages"][1]["tool_calls"][0]
    
    # User message should be clean
    assert sanitized_mistral["messages"][2]["role"] == "user"
    assert "cache_control" not in sanitized_mistral["messages"][2]

    # 3. Sanitize for Gemini (is_gemini=True)
    openai_req_gemini = anthropic_to_openai_request(anth_req, "gemini-model")
    sanitized_gemini = sanitize_openai_payload(openai_req_gemini, is_gemini=True)
    
    # Assistant message should have extra_content PRESERVED
    assert sanitized_gemini["messages"][1]["role"] == "assistant"
    assert "extra_content" in sanitized_gemini["messages"][1]["tool_calls"][0]
    assert sanitized_gemini["messages"][1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG_STUFF"

@pytest.mark.anyio
async def test_rate_limiter_logic():
    # Import from core.rate_limiter
    from core.rate_limiter import RateLimiter
    import asyncio

    # Test 2 TPS (0.5s interval)
    limiter = RateLimiter(2.0)

    start = time.time()
    # First call - should be instant
    await limiter.wait()
    # Second call - should wait ~0.5s
    await limiter.wait()
    # Third call - should wait ~0.5s (total ~1.0s)
    await limiter.wait()
    end = time.time()

    duration = end - start
    # We expect roughly 1.0s total wait for 3 calls at 2 TPS
    assert 0.9 <= duration <= 1.2

    # Test disabled (0 TPS)
    limiter.set_rate(0)
    start = time.time()
    for _ in range(10):
        await limiter.wait()
    assert time.time() - start < 0.1

    # Should be near instant
    assert (end - start) < 0.1

@pytest.mark.anyio
async def test_per_provider_rate_limiter():
    from core.rate_limiter import PerProviderRateLimiter
    import asyncio

    pl = PerProviderRateLimiter(global_tps=0)

    # Provider A: 2 TPS (0.5s interval)
    # Provider B: 2 TPS (0.5s interval)
    # They should be independent

    start = time.time()

    # Two requests to provider A back-to-back
    await pl.wait_for_provider(1, 2.0)
    await pl.wait_for_provider(1, 2.0)
    a_elapsed = time.time() - start

    start = time.time()
    # Two requests to provider B back-to-back
    await pl.wait_for_provider(2, 2.0)
    await pl.wait_for_provider(2, 2.0)
    b_elapsed = time.time() - start

    # Each pair should take ~0.5s (2 TPS = 0.5s interval)
    assert 0.4 <= a_elapsed <= 0.7
    assert 0.4 <= b_elapsed <= 0.7

    # Now test that cross-provider calls don't block each other
    # Reset timing
    pl = PerProviderRateLimiter(global_tps=0)
    start = time.time()
    await pl.wait_for_provider(1, 2.0)  # instant
    await pl.wait_for_provider(2, 2.0)  # should be instant (different provider)
    cross_elapsed = time.time() - start
    assert cross_elapsed < 0.1  # no blocking between providers

    # Test global fallback (no per-provider limit)
    pl2 = PerProviderRateLimiter(global_tps=2.0)
    start = time.time()
    await pl2.wait_for_provider(99, None)  # uses global
    await pl2.wait_for_provider(99, None)  # uses global
    global_elapsed = time.time() - start
    assert 0.4 <= global_elapsed <= 0.7  # 2 TPS = 0.5s interval

# --- Unit Tests for DB Limits and Logging ---

def test_database_logging_limits(tmp_path):
    # Override database path for isolated test
    import os
    db_file = os.path.join(tmp_path, "test_proxy.db")
    db.DB_PATH = db_file
    
    db.init_db()
    
    # Ensure starting empty
    db.clear_logs()
    assert len(db.get_logs()) == 0
    
    # Set limit to 3 logs
    db.set_log_limit(3)
    assert db.get_log_limit() == 3
    
    # Insert 5 logs
    for i in range(5):
        db.add_log(
            provider_name=f"Provider {i}",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body="{}",
            response_status=200,
            response_body=f"Response {i}"
        )
        # Add a tiny sleep to guarantee monotonic timestamps order
        time.sleep(0.01)
        
    logs = db.get_logs()
    # Should be capped at 3
    assert len(logs) == 3
    
    # Should keep the newest ones (2, 3, 4)
    provider_names = [l["provider_name"] for l in logs]
    assert "Provider 4" in provider_names
    assert "Provider 3" in provider_names
    assert "Provider 2" in provider_names
    assert "Provider 0" not in provider_names
    
    # Set limit to -1 (disabled)
    db.set_log_limit(-1)
    db.add_log(
        provider_name="Should Not Log",
        request_method="POST",
        request_path="/v1/chat/completions",
        request_body="{}",
        response_status=200,
        response_body="No-op"
    )
    # Check that database has 0 logs now
    assert len(db.get_logs()) == 0

# --- Unit Tests for Model Routing ---

def test_model_mapping_db(tmp_path):
    import os
    db_file = os.path.join(tmp_path, "test_routing.db")
    db.DB_PATH = db_file
    db.init_db()

    # Create two providers
    db.add_provider("P1", "openai", "url1", "key1", "model1", is_active=0)
    db.add_provider("P2", "openai", "url2", "key2", "model2", is_active=0)

    providers = db.get_providers()
    p1_id = providers[0]["id"]
    p2_id = providers[1]["id"]

    # Map a custom model ID to P1
    db.add_model_mapping("custom-model-a", p1_id)
    # Map another custom model ID to P1 (mix and match)
    db.add_model_mapping("custom-model-b", p1_id)
    # Map a third to P2
    db.add_model_mapping("custom-model-c", p2_id)

    mappings = db.get_model_mappings()
    assert len(mappings) == 3

    # Verify a specific mapping
    mapping_a = next(m for m in mappings if m["model_id"] == "custom-model-a")
    assert mapping_a["provider_id"] == p1_id
    assert mapping_a["provider_name"] == "P1"

def test_provider_service_model_lookup(tmp_path):
    import os
    db_file = os.path.join(tmp_path, "test_lookup.db")
    db.DB_PATH = db_file
    db.init_db()

    db.add_provider("P1", "openai", "url1", "key1", "model1", is_active=0)
    providers = db.get_providers()
    p1_id = providers[0]["id"]
    db.add_model_mapping("my-special-model", p1_id)

    from core.providers.service import ProviderService
    service = ProviderService()
    provider = service.get_provider_by_model("my-special-model")

    assert provider is not None
    assert provider.name == "P1"

    # Test non-existent model
    assert service.get_provider_by_model("ghost-model") is None

@pytest.mark.anyio
async def test_router_routing_logic(tmp_path):
    import os
    db_file = os.path.join(tmp_path, "test_router.db")
    db.DB_PATH = db_file
    db.init_db()

    # Setup providers
    db.add_provider("GlobalProvider", "openai", "url-global", "key-global", "model-global", is_active=1)
    db.add_provider("SpecialProvider", "openai", "url-special", "key-special", "model-special", is_active=0)

    providers = db.get_providers()
    global_p = next(p for p in providers if p["name"] == "GlobalProvider")
    special_p = next(p for p in providers if p["name"] == "SpecialProvider")

    # Map "special-model-id" to SpecialProvider
    db.add_model_mapping("special-model-id", special_p["id"])

    # Initialize RouterService
    async with httpx.AsyncClient() as client:
        from core.rate_limiter import RateLimiter
        from core.router import RouterService
        router = RouterService(http_client=client, rate_limiter=RateLimiter(100))
        # Invalidate cache to ensure it reads from the new DB
        router.provider_service.reload_active_provider()

        # Verify the provider selection logic used by the router
        service = router.provider_service
        assert service.get_provider_by_model("special-model-id").name == "SpecialProvider"

        # Request with unmapped model ID -> should fall back to GlobalProvider (conceptually)
        assert service.get_provider_by_model("unknown-model") is None
        assert service.get_active_provider().name == "GlobalProvider"

# --- Unit Tests for Max Tokens ---

def test_database_max_tokens_default(tmp_path):
    """Test that global max_tokens defaults to 32000."""
    import os
    db_file = os.path.join(tmp_path, "test_max_tokens.db")
    db.DB_PATH = db_file
    db.init_db()
    assert db.get_max_tokens() == 32000

def test_database_max_tokens_set_get(tmp_path):
    """Test setting and getting global max_tokens."""
    import os
    db_file = os.path.join(tmp_path, "test_max_tokens2.db")
    db.DB_PATH = db_file
    db.init_db()
    db.set_max_tokens(16000)
    assert db.get_max_tokens() == 16000
    db.set_max_tokens(8192)
    assert db.get_max_tokens() == 8192

def test_provider_max_tokens_passthrough(tmp_path):
    """Test that max_tokens is stored and passed through to providers."""
    import os
    db_file = os.path.join(tmp_path, "test_provider_max_tokens.db")
    db.DB_PATH = db_file
    db.init_db()

    # Add provider with max_tokens
    db.add_provider("TestMaxTokens", "openai", "https://api.test.com/v1/chat/completions",
                     "sk-test", "test-model", is_active=0, max_tokens=8192)
    providers = db.get_providers()
    assert providers[0]["max_tokens"] == 8192

    # Add provider without max_tokens
    db.add_provider("TestNoMaxTokens", "openai", "https://api.test.com/v1/chat/completions",
                     "sk-test", "test-model", is_active=0)
    providers = db.get_providers()
    assert providers[1]["max_tokens"] is None

def test_provider_max_tokens_update(tmp_path):
    """Test updating provider max_tokens."""
    import os
    db_file = os.path.join(tmp_path, "test_update_max_tokens.db")
    db.DB_PATH = db_file
    db.init_db()

    db.add_provider("ToUpdate", "openai", "url", "key", "model", is_active=0, max_tokens=4096)
    providers = db.get_providers()
    pid = providers[0]["id"]

    db.update_provider(pid, "ToUpdate", "openai", "url", "key", "model", 0, max_tokens=16384)
    updated = [p for p in db.get_providers() if p["id"] == pid][0]
    assert updated["max_tokens"] == 16384

def test_provider_factory_passes_max_tokens():
    """Test that ProviderFactory passes max_tokens to provider instances."""
    factory = ProviderFactory()
    config = {
        "name": "MaxTokensProvider",
        "api_type": "openai",
        "endpoint_url": "https://api.test.com/v1/chat/completions",
        "api_key": "sk-test",
        "model_name": "test-model",
        "is_active": 0,
        "id": 1,
        "max_tokens": 8192
    }
    provider = factory.create_provider(config)
    assert provider.max_tokens == 8192

def test_provider_factory_none_max_tokens():
    """Test that ProviderFactory handles None max_tokens."""
    factory = ProviderFactory()
    config = {
        "name": "NoMaxTokens",
        "api_type": "openai",
        "endpoint_url": "https://api.test.com/v1/chat/completions",
        "api_key": "sk-test",
        "model_name": "test-model",
        "is_active": 0,
        "id": 1
    }
    provider = factory.create_provider(config)
    assert provider.max_tokens is None

def test_openai_response_null_usage():
    """Test that openai_to_anthropic_response handles null usage (e.g. NVIDIA NIM)."""
    openai_res = {
        "id": "chatcmpl-nvidia",
        "model": "minimaxai/minimax-m3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": None
    }
    anth_res = openai_to_anthropic_response(openai_res)
    assert anth_res["content"][0]["text"] == "Hello! How can I help you?"
    assert anth_res["usage"]["input_tokens"] == 0
    assert anth_res["usage"]["output_tokens"] == 0

def test_openai_response_missing_usage():
    """Test that openai_to_anthropic_response handles missing usage key."""
    openai_res = {
        "id": "chatcmpl-nousage",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response text"
                },
                "finish_reason": "stop"
            }
        ]
    }
    anth_res = openai_to_anthropic_response(openai_res)
    assert anth_res["usage"]["input_tokens"] == 0
    assert anth_res["usage"]["output_tokens"] == 0

def test_settings_request_max_tokens():
    """Test SettingsRequest model accepts max_tokens."""
    from models.request import SettingsRequest
    req = SettingsRequest(max_tokens=16000)
    assert req.max_tokens == 16000

    req2 = SettingsRequest()
    assert req2.max_tokens is None

def test_settings_response_max_tokens():
    """Test SettingsResponse model includes max_tokens."""
    from models.provider import SettingsResponse
    resp = SettingsResponse(log_limit=50, rate_limit_tps=0.0, max_tokens=32000)
    assert resp.max_tokens == 32000

# --- Unit Tests for Image Content ---

# Minimal valid base64-encoded 1x1 white pixel PNG
TINY_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

def test_anthropic_to_openai_request_with_image():
    """Test that Anthropic image blocks are converted to OpenAI image_url blocks."""
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": TINY_BASE64
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")

    msg = openai_req["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2

    # Text block preserved
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][0]["text"] == "What's in this image?"

    # Image block converted
    img = msg["content"][1]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"] == f"data:image/png;base64,{TINY_BASE64}"


def test_anthropic_to_openai_request_image_only():
    """Test Anthropic message with only an image (no text) produces content array."""
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": TINY_BASE64
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512
    }

    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")

    msg = openai_req["messages"][0]
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "image_url"
    assert "data:image/jpeg;base64," in msg["content"][0]["image_url"]["url"]


def test_anthropic_to_openai_request_multiple_images():
    """Test multiple images in a single message."""
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": TINY_BASE64}
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/gif", "data": TINY_BASE64}
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")

    content = openai_req["messages"][0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"
    assert "data:image/png;base64," in content[1]["image_url"]["url"]
    assert "data:image/gif;base64," in content[2]["image_url"]["url"]


def test_openai_to_anthropic_request_with_image():
    """Test that OpenAI image_url blocks are converted to Anthropic image blocks."""
    openai_req = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{TINY_BASE64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    anth_req = openai_to_anthropic_request(openai_req, "claude-3-5-sonnet")

    msg = anth_req["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2

    # Text block preserved
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][0]["text"] == "Describe this image"

    # Image block converted
    img = msg["content"][1]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/jpeg"
    assert img["source"]["data"] == TINY_BASE64


def test_openai_to_anthropic_request_image_only():
    """Test OpenAI message with only an image_url produces content array."""
    openai_req = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TINY_BASE64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512
    }

    anth_req = openai_to_anthropic_request(openai_req, "claude-3-5-sonnet")

    msg = anth_req["messages"][0]
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "image"
    assert msg["content"][0]["source"]["media_type"] == "image/png"


def test_openai_to_anthropic_request_multiple_images():
    """Test multiple image_url blocks in a single OpenAI message."""
    openai_req = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_BASE64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{TINY_BASE64}"}}
                ]
            }
        ],
        "max_tokens": 1024
    }

    anth_req = openai_to_anthropic_request(openai_req, "claude-3-5-sonnet")

    content = anth_req["messages"][0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[2]["type"] == "image"
    assert content[2]["source"]["media_type"] == "image/webp"


def test_roundtrip_image_preserves_data():
    """Test that Anthropic -> OpenAI -> Anthropic roundtrip preserves image data."""
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": TINY_BASE64}
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")
    roundtrip = openai_to_anthropic_request(openai_req, "claude-3-5-sonnet")

    msg = roundtrip["messages"][0]
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][0]["text"] == "Look"
    img = msg["content"][1]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == TINY_BASE64


def test_sanitize_preserves_image_url():
    """Test that sanitize_openai_payload preserves image_url content blocks."""
    openai_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_BASE64}"}}
                ]
            }
        ]
    }

    sanitized = sanitize_openai_payload(openai_payload, is_gemini=False)

    content = sanitized["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{TINY_BASE64}"


def test_sanitize_flattens_text_only():
    """Test that sanitize still flattens text-only content to string."""
    openai_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }
        ]
    }

    sanitized = sanitize_openai_payload(openai_payload, is_gemini=False)

    content = sanitized["messages"][0]["content"]
    assert isinstance(content, str)
    assert content == "HelloWorld"

@pytest.mark.xfail(reason="Anthropic tool_result with image translation - requires dedicated work on translation layer")
def test_anthropic_tool_result_with_image():
    """Test that Anthropic tool_result with image content preserves images for OpenAI providers."""
    anth_req = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that file"},
                    {
                        "type": "tool_use",
                        "id": "toolu_read_123",
                        "name": "Read",
                        "input": {"file_path": "/tmp/image.jpg"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_123",
                        "content": [
                            {"type": "text", "text": "File contents:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": TINY_BASE64
                                }
                            }
                        ]
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    openai_req = anthropic_to_openai_request(anth_req, "gpt-4o")

    tool_msg = openai_req["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "toolu_read_123"
    # Content should preserve text and image blocks
    assert isinstance(tool_msg["content"], list)
    assert len(tool_msg["content"]) == 2
    assert tool_msg["content"][0]["type"] == "text"
    assert tool_msg["content"][0]["text"] == "File contents:"
    assert tool_msg["content"][1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in tool_msg["content"][1]["image_url"]["url"]

