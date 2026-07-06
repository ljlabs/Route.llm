import pytest
import json
import time
import os
import sys
import subprocess
import database as db
import httpx
from core.providers.factory import ProviderFactory
from core.providers.embedding import EmbeddingProvider

# --- Unit Tests for EmbeddingProvider ---

def test_embedding_provider_factory():
    """ProviderFactory creates EmbeddingProvider from config."""
    config = {
        "name": "Test Embedding",
        "api_type": "embedding",
        "endpoint_url": "https://generativelanguage.googleapis.com/v1beta/openai/v1/embeddings",
        "api_key": "ai-test-key",
        "model_name": "gemini-embedding-2",
        "is_active": 0,
        "id": 99
    }
    provider = ProviderFactory.create_provider(config)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.api_type == "embedding"
    assert provider.name == "Test Embedding"
    assert provider.model_name == "gemini-embedding-2"


def test_embedding_wrap_request():
    """wrap_request substitutes model_name and passes input through."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="key",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    req = {"model": "some-model", "input": "Hello world"}
    wrapped = provider.wrap_request(req)
    assert wrapped["model"] == "gemini-embedding-2"
    assert wrapped["input"] == "Hello world"


def test_embedding_wrap_request_list():
    """wrap_request handles list input."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="key",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    req = {"model": "some-model", "input": ["Hello", "World"]}
    wrapped = provider.wrap_request(req)
    assert wrapped["input"] == ["Hello", "World"]
    assert wrapped["model"] == "gemini-embedding-2"


def test_embedding_unwrap_response():
    """unwrap_response is passthrough."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="key",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    response = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "gemini-embedding-2",
        "usage": {"prompt_tokens": 8, "total_tokens": 8}
    }
    result = provider.unwrap_response(response)
    assert result == response


def test_embedding_headers():
    """Headers include Authorization with Bearer token."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="test-key-123",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    headers = provider.get_headers()
    assert headers["Authorization"] == "Bearer test-key-123"
    assert headers["Content-Type"] == "application/json"


def test_embedding_requires_translation():
    """Embedding provider does not require translation."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="key",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    assert provider.requires_translation() is False


def test_embedding_stream_translator():
    """Embedding provider returns None for stream translator."""
    provider = EmbeddingProvider(
        name="Test", endpoint_url="http://test.com", api_key="key",
        model_name="gemini-embedding-2", api_type="embedding"
    )
    assert provider.get_stream_translator() is None


def test_embedding_request_model():
    """EmbeddingRequest validates correctly."""
    from models.request import EmbeddingRequest

    req = EmbeddingRequest(model="gemini-embedding-2", input="Hello world")
    data = req.model_dump()
    assert data["model"] == "gemini-embedding-2"
    assert data["input"] == "Hello world"
    assert data["encoding_format"] == "float"

    # Test with list input
    req2 = EmbeddingRequest(model="gemini-embedding-2", input=["Hello", "World"])
    data2 = req2.model_dump()
    assert data2["input"] == ["Hello", "World"]


# --- Database Tests ---

def test_embedding_provider_in_db(tmp_path):
    """Embedding provider can be stored and retrieved from the database."""
    db_file = os.path.join(tmp_path, "test_embedding.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    db.add_provider(
        name="Google AI Studio Embeddings",
        api_type="embedding",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/openai/v1/embeddings",
        api_key="ai-test-key",
        model_name="gemini-embedding-2",
        is_active=1
    )

    providers = db.get_providers()
    assert len(providers) == 1
    assert providers[0]["api_type"] == "embedding"
    assert providers[0]["model_name"] == "gemini-embedding-2"

    # Verify active lookup works
    active = db.get_active_provider()
    assert active is not None
    assert active["api_type"] == "embedding"

    db.DB_PATH = os.path.join(os.path.dirname(db.__file__), "proxy.db")


# --- Integration Tests ---

@pytest.mark.anyio
async def test_embedding_no_provider_returns_400():
    """Embedding endpoint returns 400 when no provider is configured."""
    from core.embedding.router import EmbeddingRouterService
    from core.rate_limiter import PerProviderRateLimiter
    from infrastructure.http_client import init_http_client
    from fastapi import HTTPException
    
    # Create a router service with no providers
    http_client = init_http_client()
    router_service = EmbeddingRouterService(http_client, PerProviderRateLimiter())
    
    # Mock get_active_embedding_provider to return None
    original_get = db.get_active_embedding_provider
    db.get_active_embedding_provider = lambda: None
    
    try:
        with pytest.raises(HTTPException) as exc_info:
            await router_service.route_embedding_request({
                "model": "unknown",
                "input": "test"
            })
        assert exc_info.value.status_code == 400
        assert "No active embedding provider" in exc_info.value.detail
    finally:
        db.get_active_embedding_provider = original_get
        await http_client.close()


@pytest.mark.anyio
async def test_embedding_provider_crud():
    """Embedding provider CRUD operations work correctly."""
    # Create temporary database
    temp_db = "test_embedding_crud.db"
    original_db = db.DB_PATH
    db.DB_PATH = temp_db
    
    try:
        db.init_db()
        
        # Test CREATE
        db.add_provider(
            name="Test Embedding 1",
            api_type="embedding",
            endpoint_url="http://test1.com/embeddings",
            api_key="key1",
            model_name="model1",
            is_active=0
        )
        
        providers = db.get_providers()
        assert len(providers) >= 1
        test_prov = [p for p in providers if p["name"] == "Test Embedding 1"][0]
        assert test_prov["model_name"] == "model1"
        
        # Test UPDATE
        db.update_provider(
            test_prov["id"],
            name="Test Embedding Updated",
            api_type="embedding",
            endpoint_url="http://test1-updated.com/embeddings",
            api_key="key1-updated",
            model_name="model1-updated",
            is_active=0
        )
        
        providers = db.get_providers()
        updated = [p for p in providers if p["id"] == test_prov["id"]][0]
        assert updated["name"] == "Test Embedding Updated"
        assert updated["model_name"] == "model1-updated"
        
        # Test DELETE
        db.delete_provider(test_prov["id"])
        providers = db.get_providers()
        deleted = [p for p in providers if p["id"] == test_prov["id"]]
        assert len(deleted) == 0
        
    finally:
        db.DB_PATH = original_db
        if os.path.exists(temp_db):
            os.remove(temp_db)


@pytest.mark.anyio
async def test_embedding_route_with_provider():
    """Embedding request routing and logging works correctly."""
    from core.embedding.router import EmbeddingRouterService
    from core.rate_limiter import PerProviderRateLimiter
    from infrastructure.http_client import init_http_client
    from unittest.mock import AsyncMock, MagicMock
    
    # Create temporary database with provider
    temp_db = "test_embedding_route.db"
    original_db = db.DB_PATH
    db.DB_PATH = temp_db
    
    try:
        db.init_db()
        db.clear_logs()
        
        # Add a test provider
        db.add_provider(
            name="Test Embedding Provider",
            api_type="embedding",
            endpoint_url="http://mock-embedding.test/v1/embeddings",
            api_key="test-key",
            model_name="test-embedding-model",
            is_active=0
        )
        
        # Get provider and set as active embedding
        providers = db.get_providers()
        provider = [p for p in providers if p["name"] == "Test Embedding Provider"][0]
        db.set_active_embedding_provider(provider["id"])
        
        # Create router with mocked HTTP client
        http_client = AsyncMock()
        http_client.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            text='{"data": [{"embedding": [0.1, 0.2], "index": 0}], "usage": {"prompt_tokens": 5, "total_tokens": 5}, "model": "test-embedding-model"}',
            json=MagicMock(return_value={
                "data": [{"embedding": [0.1, 0.2], "index": 0}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
                "model": "test-embedding-model"
            })
        ))
        
        router_service = EmbeddingRouterService(http_client, PerProviderRateLimiter())
        
        # Make request
        response = await router_service.route_embedding_request({
            "model": "test-embedding-model",
            "input": "test embedding"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.body
        assert b"embedding" in data
        
        # Verify logging occurred
        logs = db.get_logs()
        assert len(logs) > 0
        request_logs = [l for l in logs if l["request_path"] == "/v1/embeddings"]
        assert len(request_logs) > 0
        
    finally:
        db.DB_PATH = original_db
        if os.path.exists(temp_db):
            os.remove(temp_db)



# --- Unit Tests for NvidiaNimEmbeddingProvider ---

from core.providers.nvidia_nim import NvidiaNimEmbeddingProvider


def _nim_provider():
    return NvidiaNimEmbeddingProvider(
        name="NIM Test",
        endpoint_url="https://integrate.api.nvidia.com/v1/embeddings",
        api_key="nim-key-123",
        model_name="nvidia/nv-embedcode-7b-v1",
    )


def test_nim_factory():
    """ProviderFactory creates NvidiaNimEmbeddingProvider from config."""
    config = {
        "name": "NIM Embed",
        "api_type": "embedding_nvidia_nim",
        "endpoint_url": "https://integrate.api.nvidia.com/v1/embeddings",
        "api_key": "nim-key",
        "model_name": "nvidia/nv-embedcode-7b-v1",
        "is_active": 0,
        "id": 42,
    }
    provider = ProviderFactory.create_provider(config)
    assert isinstance(provider, NvidiaNimEmbeddingProvider)
    assert provider.api_type == "embedding_nvidia_nim"


def test_nim_wrap_request_defaults_input_type_to_query():
    """input_type defaults to 'query' when not supplied by client."""
    provider = _nim_provider()
    wrapped = provider.wrap_request({"input": "What is Python?"})
    assert wrapped["input_type"] == "query"


def test_nim_wrap_request_respects_client_input_type():
    """input_type from client overrides the default."""
    provider = _nim_provider()
    wrapped = provider.wrap_request({"input": "Some passage text", "input_type": "passage"})
    assert wrapped["input_type"] == "passage"


def test_nim_wrap_request_model_override():
    """wrap_request always uses provider's model_name, not client-supplied model."""
    provider = _nim_provider()
    wrapped = provider.wrap_request({"input": "hello", "model": "some-other-model"})
    assert wrapped["model"] == "nvidia/nv-embedcode-7b-v1"


def test_nim_wrap_request_passes_encoding_format_and_truncate():
    """encoding_format and truncate are forwarded when present."""
    provider = _nim_provider()
    wrapped = provider.wrap_request({
        "input": "hello",
        "encoding_format": "float",
        "truncate": "END",
    })
    assert wrapped["encoding_format"] == "float"
    assert wrapped["truncate"] == "END"


def test_nim_wrap_request_omits_absent_optional_fields():
    """encoding_format and truncate are absent from payload when not supplied."""
    provider = _nim_provider()
    wrapped = provider.wrap_request({"input": "hello"})
    assert "encoding_format" not in wrapped
    assert "truncate" not in wrapped


def test_nim_headers():
    """Headers include Bearer token and correct content type."""
    provider = _nim_provider()
    headers = provider.get_headers()
    assert headers["Authorization"] == "Bearer nim-key-123"
    assert headers["Content-Type"] == "application/json"


def test_nim_requires_translation_false():
    assert _nim_provider().requires_translation() is False


def test_nim_stream_translator_none():
    assert _nim_provider().get_stream_translator() is None


def test_nim_unwrap_response_passthrough():
    """unwrap_response is a passthrough."""
    provider = _nim_provider()
    resp = {"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 5}}
    assert provider.unwrap_response(resp) == resp


def test_embedding_request_model_input_type_field():
    """EmbeddingRequest accepts input_type and truncate fields."""
    from models.request import EmbeddingRequest

    req = EmbeddingRequest(
        model="nvidia/nv-embedcode-7b-v1",
        input="What is the capital of France?",
        input_type="query",
        truncate="NONE",
        encoding_format="float",
    )
    data = req.model_dump(exclude_none=True)
    assert data["input_type"] == "query"
    assert data["truncate"] == "NONE"
    assert data["encoding_format"] == "float"


def test_embedding_request_input_type_defaults_none():
    """EmbeddingRequest input_type is None when not supplied (NIM provider applies its own default)."""
    from models.request import EmbeddingRequest

    req = EmbeddingRequest(model="nvidia/nv-embedcode-7b-v1", input="hello")
    assert req.input_type is None
