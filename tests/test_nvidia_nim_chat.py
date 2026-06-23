"""
Unit tests for NvidiaNimChatProvider.

Covers:
- Factory instantiation with api_type="nvidia_nim"
- Default hyperparameters are injected when absent from client request
- Client-supplied values override every default (individually and collectively)
- Fields from the Anthropic→OpenAI translation that overlap with NIM defaults
  (max_tokens, temperature, top_p) honour client values
- Non-hyperparameter fields (messages, model, stream, tools) are unaffected
- Headers include Bearer token
- requires_translation() returns True (inherits OpenAI behaviour)
- get_stream_translator() returns a valid translator
- End-to-end: RouterService sends NIM defaults to the upstream provider
"""

import json
import os
import pytest
import httpx
import respx

import database as db
from core.providers.nvidia_nim import NvidiaNimChatProvider, _NIM_DEFAULTS
from core.providers.factory import ProviderFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(**kwargs) -> NvidiaNimChatProvider:
    defaults = dict(
        name="NIM Chat Test",
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key="nvapi-test-key",
        model_name="meta/llama-3.3-70b-instruct",
    )
    defaults.update(kwargs)
    return NvidiaNimChatProvider(**defaults)


def _simple_anthropic_request(**overrides) -> dict:
    """Minimal Anthropic-format chat request."""
    req = {
        "messages": [{"role": "user", "content": "Hello"}],
    }
    req.update(overrides)
    return req


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_creates_nim_chat_provider():
    """ProviderFactory returns NvidiaNimChatProvider for api_type='nvidia_nim'."""
    config = {
        "name": "NIM Chat",
        "api_type": "nvidia_nim",
        "endpoint_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "api_key": "nvapi-key",
        "model_name": "meta/llama-3.3-70b-instruct",
        "is_active": 1,
        "id": 99,
    }
    provider = ProviderFactory.create_provider(config)
    assert isinstance(provider, NvidiaNimChatProvider)
    assert provider.api_type == "nvidia_nim"
    assert provider.model_name == "meta/llama-3.3-70b-instruct"


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

def test_all_nim_defaults_injected_when_absent():
    """All six NIM defaults appear in the payload when the client sends none."""
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())

    for key, expected in _NIM_DEFAULTS.items():
        assert key in payload, f"Expected default '{key}' to be present"
        assert payload[key] == expected, f"'{key}': expected {expected}, got {payload[key]}"


def test_default_max_tokens():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["max_tokens"] == 4096


def test_default_temperature():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["temperature"] == 0.1


def test_default_top_p():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["top_p"] == 0.9


def test_default_repetition_penalty():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["repetition_penalty"] == 1.12


def test_default_frequency_penalty():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["frequency_penalty"] == 0.2


def test_default_presence_penalty():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["presence_penalty"] == 0.05


# ---------------------------------------------------------------------------
# Client overrides
# ---------------------------------------------------------------------------

def test_client_max_tokens_overrides_default():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request(max_tokens=512))
    assert payload["max_tokens"] == 512


def test_client_temperature_overrides_default():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request(temperature=0.9))
    assert payload["temperature"] == 0.9


def test_client_top_p_overrides_default():
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request(top_p=0.5))
    assert payload["top_p"] == 0.5


@pytest.mark.parametrize("key,value", [
    ("repetition_penalty", 1.0),
    ("frequency_penalty", 0.0),
    ("presence_penalty", 0.5),
])
def test_client_nim_specific_param_overrides_default(key, value):
    """NIM-specific params passed at the top level of the Anthropic request are honoured."""
    provider = _make_provider()
    # These params aren't standard Anthropic fields; they pass through as extra keys
    req = _simple_anthropic_request(**{key: value})
    payload = provider.wrap_request(req)
    # The override must be respected — default must NOT clobber it
    assert payload[key] == value, f"Client value for '{key}' should override default"


def test_all_defaults_overridden_simultaneously():
    """A client that sets every hyperparameter gets exactly what it asked for."""
    provider = _make_provider()
    overrides = {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }
    payload = provider.wrap_request(_simple_anthropic_request(**overrides))
    for key, expected in overrides.items():
        assert payload[key] == expected, f"'{key}' mismatch"


# ---------------------------------------------------------------------------
# Request structure
# ---------------------------------------------------------------------------

def test_model_name_is_set_from_provider_config():
    """The provider's model_name is used, not anything in the raw request."""
    provider = _make_provider(model_name="meta/llama-3.3-70b-instruct")
    payload = provider.wrap_request(_simple_anthropic_request())
    assert payload["model"] == "meta/llama-3.3-70b-instruct"


def test_messages_are_translated():
    """Messages appear in the payload after Anthropic→OpenAI translation."""
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    assert "messages" in payload
    assert any(m["role"] == "user" for m in payload["messages"])


def test_stream_flag_passed_through():
    """stream=True is preserved in the payload."""
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request(stream=True))
    assert payload.get("stream") is True


def test_system_prompt_included():
    """A system prompt is translated into a system message."""
    provider = _make_provider()
    req = _simple_anthropic_request(system="You are a helpful assistant.")
    payload = provider.wrap_request(req)
    system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
    assert system_msgs, "Expected a system message in the translated payload"
    assert "helpful assistant" in system_msgs[0]["content"]


def test_nim_defaults_do_not_bleed_into_messages():
    """Hyperparameter keys are not injected into the messages list."""
    provider = _make_provider()
    payload = provider.wrap_request(_simple_anthropic_request())
    for msg in payload["messages"]:
        for key in _NIM_DEFAULTS:
            assert key not in msg, f"Hyperparameter '{key}' should not appear inside a message"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def test_headers_contain_bearer_token():
    provider = _make_provider(api_key="nvapi-secret")
    headers = provider.get_headers()
    assert headers["Authorization"] == "Bearer nvapi-secret"
    assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Translation / streaming
# ---------------------------------------------------------------------------

def test_requires_translation_true():
    """NIM chat provider requires Anthropic↔OpenAI translation (inherits OpenAI)."""
    assert _make_provider().requires_translation() is True


def test_get_stream_translator_returns_object():
    """get_stream_translator returns a non-None translator for 'anthropic' target."""
    translator = _make_provider().get_stream_translator("anthropic")
    assert translator is not None


# ---------------------------------------------------------------------------
# End-to-end: RouterService → NIM upstream
#
# These tests wire up a real RouterService backed by an in-memory DB, register
# a NIM provider as the active provider, and intercept the outbound HTTP call
# with respx.  We then assert on the *exact payload* that would be sent to
# NVIDIA NIM — confirming that the defaults travel all the way through the
# router, not just through wrap_request in isolation.
# ---------------------------------------------------------------------------

_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Minimal OpenAI-format response that satisfies the router's response handling
_FAKE_NIM_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "meta/llama-3.3-70b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _setup_nim_db(tmp_path) -> None:
    """Point the shared DB module at a fresh temp DB and add a NIM provider."""
    db.DB_PATH = str(tmp_path / "nim_e2e.db")
    db.init_db()
    db.add_provider(
        name="NIM E2E",
        api_type="nvidia_nim",
        endpoint_url=_NIM_ENDPOINT,
        api_key="nvapi-test",
        model_name="meta/llama-3.3-70b-instruct",
        is_active=1,
    )


@pytest.mark.anyio
async def test_router_sends_nim_defaults_to_upstream(tmp_path):
    """
    A bare Anthropic request (no hyperparams) reaching the RouterService should
    result in an outbound payload to NIM that contains all six default values.
    """
    _setup_nim_db(tmp_path)

    captured: dict = {}

    def capture_and_respond(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_FAKE_NIM_RESPONSE)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(_NIM_ENDPOINT).mock(side_effect=capture_and_respond)

        async with httpx.AsyncClient() as client:
            from core.rate_limiter import RateLimiter
            from core.router import RouterService
            from core.providers.service import ProviderService

            # Fresh service so it reads our temp DB
            router = RouterService(http_client=client, rate_limiter=RateLimiter(0))
            router.provider_service = ProviderService()

            anthropic_request = {
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": "Hello NIM"}],
                "max_tokens": 4096,  # required by the router; matches NIM default
            }
            await router.route_anthropic_request(anthropic_request, stream=False)

    payload = captured["body"]

    for key, expected in _NIM_DEFAULTS.items():
        assert key in payload, f"Expected '{key}' in outbound NIM payload"
        assert payload[key] == expected, (
            f"'{key}': expected default {expected!r}, got {payload[key]!r}"
        )


@pytest.mark.anyio
async def test_router_respects_client_hyperparams_over_nim_defaults(tmp_path):
    """
    When the client explicitly sets hyperparameters in the Anthropic request,
    those values — not the NIM defaults — must reach the upstream provider.
    """
    _setup_nim_db(tmp_path)

    captured: dict = {}

    def capture_and_respond(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_FAKE_NIM_RESPONSE)

    client_overrides = {
        "max_tokens": 512,
        "temperature": 0.9,
        "top_p": 0.5,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.3,
    }

    with respx.mock(assert_all_called=False) as mock:
        mock.post(_NIM_ENDPOINT).mock(side_effect=capture_and_respond)

        async with httpx.AsyncClient() as client:
            from core.rate_limiter import RateLimiter
            from core.router import RouterService
            from core.providers.service import ProviderService

            router = RouterService(http_client=client, rate_limiter=RateLimiter(0))
            router.provider_service = ProviderService()

            anthropic_request = {
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": "Override test"}],
                **client_overrides,
            }
            await router.route_anthropic_request(anthropic_request, stream=False)

    payload = captured["body"]

    for key, expected in client_overrides.items():
        assert payload[key] == expected, (
            f"'{key}': client override {expected!r} should beat NIM default, "
            f"but got {payload[key]!r}"
        )
