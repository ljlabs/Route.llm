"""
Shared pytest fixtures for the model-router conformance suite.

Configuration is via environment variables so this suite can point at
any deployment without editing code:

    BASE_URL      default: http://localhost:8001
    API_KEY       default: "test-key"
    MODEL         default: "gpt-4o-mini"   (model id your router accepts)
    ANTHROPIC_VERSION    default: "2023-06-01"

Run a subset with markers, e.g.:
    pytest -m openai
    pytest -m anthropic
    pytest -m "openai and streaming"
"""
import os
import requests
import pytest

# requests has NO default timeout -- if the server isn't listening, is
# behind a firewall that silently drops packets, or hangs mid-response,
# a call will block forever instead of failing. Every request made
# through the sessions below gets this timeout unless the call site
# passes its own. Override with REQUEST_TIMEOUT_SECONDS if your router
# is just slow (e.g. proxying to a large model) rather than broken.
DEFAULT_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))


def _env(name, default):
    return os.environ.get(name, default)


class TimeoutSession(requests.Session):
    """requests.Session that applies a default timeout to every call
    unless the caller explicitly passes one (including stream=True calls,
    where this is the connect+first-byte timeout, not a read-to-completion
    timeout)."""

    def request(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        return super().request(method, url, *args, **kwargs)


# ---------- OpenAI-spec config ----------

@pytest.fixture(scope="session")
def openai_base_url():
    return _env("BASE_URL", "http://localhost:8001").rstrip("/")


@pytest.fixture(scope="session")
def openai_api_key():
    return _env("API_KEY", "test-key")


@pytest.fixture(scope="session")
def openai_model():
    return _env("MODEL", "gpt-4o-mini")


@pytest.fixture(scope="session")
def openai_headers(openai_api_key):
    return {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="session")
def openai_session():
    s = TimeoutSession()
    yield s
    s.close()


# ---------- Anthropic-spec config ----------

@pytest.fixture(scope="session")
def anthropic_base_url():
    return _env("BASE_URL", "http://localhost:8001").rstrip("/")


@pytest.fixture(scope="session")
def anthropic_api_key():
    return _env("API_KEY", "test-key")


@pytest.fixture(scope="session")
def anthropic_model():
    return _env("MODEL", "claude-sonnet-4-6")


@pytest.fixture(scope="session")
def anthropic_version():
    return _env("ANTHROPIC_VERSION", "2023-06-01")


@pytest.fixture(scope="session")
def anthropic_headers(anthropic_api_key, anthropic_version):
    return {
        "x-api-key": anthropic_api_key,
        "anthropic-version": anthropic_version,
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="session")
def anthropic_session():
    s = TimeoutSession()
    yield s
    s.close()


def pytest_configure(config):
    config.addinivalue_line("markers", "openai: OpenAI chat-completions spec")
    config.addinivalue_line("markers", "anthropic: Anthropic messages spec")
    config.addinivalue_line("markers", "streaming: streaming (SSE) behavior")
    config.addinivalue_line("markers", "tools: tool/function calling")
    config.addinivalue_line("markers", "errors: error-format conformance")
    config.addinivalue_line("markers", "vision: image/multimodal input")