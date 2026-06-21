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

EMBEDDING_PORT = 8081
EMBEDDING_URL = f"http://127.0.0.1:{EMBEDDING_PORT}"
INTEGRATION_DB = "integ_test.db"


import atexit
import signal

# Track all spawned integration test processes for cleanup
_integ_procs: list = []

def _kill_all_integ_procs():
    """Kill all integration test server processes. Called on exit or signal."""
    for p in _integ_procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

atexit.register(_kill_all_integ_procs)

# Register signal handlers so Ctrl+C also triggers cleanup
for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, lambda s, f: (_kill_all_integ_procs(), sys.exit(1)))
    except (OSError, ValueError):
        pass  # may fail on non-main thread


@pytest.fixture(scope="module")
def embedding_server():
    """Launch embedding server on port 8081 with a fresh integ_test.db."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.path.join(project_root, INTEGRATION_DB)

    # Kill anything already on the port before starting
    with httpx.Client(timeout=1) as c:
        try:
            c.get(f"{EMBEDDING_URL}/docs")
            raise RuntimeError(f"Port {EMBEDDING_PORT} already in use — run kill_integ_servers.py to clean up")
        except Exception:
            pass

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # DB may still be locked; server will overwrite it

    env = os.environ.copy()
    env["EMBEDDING_DB_PATH"] = db_path

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "embedding_main:app",
         "--host", "127.0.0.1", "--port", str(EMBEDDING_PORT)],
        env=env,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _integ_procs.append(proc)

    # Wait for server to be ready (up to 15s)
    ready = False
    for _ in range(30):
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(f"Server exited early:\n{err.decode()}")
        try:
            with httpx.Client(timeout=1) as c:
                if c.get(f"{EMBEDDING_URL}/docs").status_code < 500:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        proc.kill()
        raise RuntimeError("Embedding server did not start in time")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    _integ_procs.remove(proc)

    # Cleanup DB
    for _ in range(5):
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            break
        except PermissionError:
            time.sleep(0.3)


@pytest.mark.anyio
async def test_embedding_no_provider_returns_400(embedding_server):
    """Embedding endpoint returns 400 when no provider is configured."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{EMBEDDING_URL}/v1/embeddings",
            json={"model": "gemini-embedding-2", "input": "Hello"}
        )
        assert resp.status_code == 400
        assert "No active embedding provider" in resp.json()["detail"]


@pytest.mark.anyio
async def test_embedding_provider_crud(embedding_server):
    """Embedding providers can be created, retrieved, and deleted via API."""
    async with httpx.AsyncClient() as client:
        # Create
        provider_data = {
            "name": "Test Embed Provider",
            "api_type": "embedding",
            "endpoint_url": f"http://127.0.0.1:{EMBEDDING_PORT}/api/providers",
            "api_key": "test-key",
            "model_name": "gemini-embedding-2",
            "is_active": 1
        }
        create_resp = await client.post(
            f"{EMBEDDING_URL}/api/providers",
            json=provider_data
        )
        assert create_resp.status_code == 200

        # List
        list_resp = await client.get(f"{EMBEDDING_URL}/api/providers")
        assert list_resp.status_code == 200
        providers = list_resp.json()
        assert len(providers) == 1
        assert providers[0]["api_type"] == "embedding"
        assert providers[0]["model_name"] == "gemini-embedding-2"

        # Cleanup
        pid = providers[0]["id"]
        del_resp = await client.delete(f"{EMBEDDING_URL}/api/providers/{pid}")
        assert del_resp.status_code == 200


@pytest.mark.anyio
async def test_embedding_route_with_provider(embedding_server):
    """Embedding request routes to provider and logs the request."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Use a non-existent local port so the router gets a fast ConnectionRefused (502).
        # This verifies routing + logging without any network dependency or self-deadlock.
        provider_data = {
            "name": "LocalEcho Embed",
            "api_type": "embedding",
            "endpoint_url": "http://127.0.0.1:19999/v1/embeddings",
            "api_key": "test-key",
            "model_name": "gemini-embedding-2",
            "is_active": 1
        }
        await client.post(f"{EMBEDDING_URL}/api/providers", json=provider_data)

        # Send embedding request
        resp = await client.post(
            f"{EMBEDDING_URL}/v1/embeddings",
            json={"model": "gemini-embedding-2", "input": "Hello world"}
        )

        # The backend is unreachable, so the router returns 502
        assert resp.status_code in (200, 400, 500, 502)

        # Check logs contain an embedding request
        logs_resp = await client.get(f"{EMBEDDING_URL}/api/logs")
        assert logs_resp.status_code == 200
        logs = logs_resp.json()
        embedding_logs = [l for l in logs if l.get("request_path") == "/v1/embeddings"]
        assert len(embedding_logs) > 0
        assert embedding_logs[0]["latency_ms"] >= 0
        assert embedding_logs[0]["provider_name"] == "LocalEcho Embed"
