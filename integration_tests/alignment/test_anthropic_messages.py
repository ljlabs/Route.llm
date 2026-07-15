"""
Conformance tests for POST /v1/messages (non-streaming) against the
Anthropic Messages API spec: https://docs.claude.com/en/api/messages
"""
import pytest
from .validators import validate_anthropic_message, validate_anthropic_error

pytestmark = pytest.mark.anthropic


def post_messages(session, base_url, headers, payload):
    return session.post(f"{base_url}/v1/messages", headers=headers, json=payload)


def test_basic_message_shape(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say the word 'pong' and nothing else."}],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_system_prompt_top_level_string(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "system": "You are a terse test fixture. Reply with one word.",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_system_prompt_as_block_array(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse test fixture."}],
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_multi_turn_conversation(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "My favorite number is 42."},
            {"role": "assistant", "content": "Got it, 42."},
            {"role": "user", "content": "What is my favorite number?"},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_content_as_block_array(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello as a content block"}]},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


@pytest.mark.parametrize("param,value", [
    ("temperature", 0.0),
    ("temperature", 1.0),
    ("top_p", 0.5),
    ("top_k", 40),
    ("stop_sequences", ["STOP"]),
    ("metadata", {"user_id": "conformance-test-user"}),
])
def test_optional_params_accepted(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model, param, value):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        param: value,
    })
    assert resp.status_code == 200, f"param {param}={value!r} -> {resp.status_code}: {resp.text}"
    validate_anthropic_message(resp.json())


def test_max_tokens_is_respected(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "Write a long paragraph about the ocean."}],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_anthropic_message(body)
    assert body["stop_reason"] == "max_tokens", (
        f"expected stop_reason='max_tokens' with tiny max_tokens, got {body['stop_reason']!r}"
    )


def test_stop_sequence_honored(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
        "stop_sequences": ["5"],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_anthropic_message(body)
    if body["stop_reason"] == "stop_sequence":
        assert body.get("stop_sequence") == "5"


# -------- Required-field / error-format conformance --------

def test_missing_max_tokens_is_400(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400, f"max_tokens is required; expected 400, got {resp.status_code}: {resp.text}"
    validate_anthropic_error(resp.json(), resp.status_code)


def test_missing_messages_is_400(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 32,
    })
    assert resp.status_code == 400, f"expected 400 for missing 'messages', got {resp.status_code}: {resp.text}"
    validate_anthropic_error(resp.json(), resp.status_code)


def test_missing_model_is_400(anthropic_session, anthropic_base_url, anthropic_headers):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400, f"expected 400 for missing 'model', got {resp.status_code}: {resp.text}"
    validate_anthropic_error(resp.json(), resp.status_code)


def test_unknown_model_is_400_or_404(anthropic_session, anthropic_base_url, anthropic_headers):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": "definitely-not-a-real-model-xyz",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code in (400, 404), f"expected 400/404 for unknown model, got {resp.status_code}: {resp.text}"
    validate_anthropic_error(resp.json(), resp.status_code)


def test_first_message_must_be_user_role(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    """Anthropic spec: conversation must start with a 'user' message (no
    system-role message inside `messages` -- that belongs in top-level
    `system` instead)."""
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "assistant", "content": "hi"}],
    })
    assert resp.status_code == 400, f"expected 400 when conversation starts with assistant role, got {resp.status_code}"
    validate_anthropic_error(resp.json(), resp.status_code)


def test_system_role_inside_messages_is_normalized(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [
            {"role": "system", "content": "Claude Code system context"},
            {"role": "user", "content": "hi"},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_missing_api_key_header_is_accepted_for_local_service(anthropic_session, anthropic_base_url, anthropic_version, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, {
        "anthropic-version": anthropic_version,
        "Content-Type": "application/json",
    }, {
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


def test_missing_anthropic_version_header(anthropic_session, anthropic_base_url, anthropic_api_key, anthropic_model):
    resp = post_messages(anthropic_session, anthropic_base_url, {
        "x-api-key": anthropic_api_key,
        "Content-Type": "application/json",
    }, {
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    })
    # Real API requires anthropic-version and 400s without it; a router may
    # choose to default it instead. Either is defensible -- it must not 500.
    assert resp.status_code < 500, f"server errored (5xx) on missing anthropic-version header: {resp.status_code}"
