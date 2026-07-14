"""
Conformance tests for POST /v1/chat/completions (non-streaming) against
the OpenAI spec: https://platform.openai.com/docs/api-reference/chat
"""
import pytest
from validators import validate_openai_chat_completion, validate_openai_error

pytestmark = pytest.mark.openai


def post_chat(session, base_url, headers, payload):
    return session.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)


def test_basic_completion_shape(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Say the word 'pong' and nothing else."}],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    assert body["model"], "model field should echo back a model id"


def test_system_message_supported(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": "You are a terse test fixture. Reply with one word."},
            {"role": "user", "content": "hello"},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())


def test_multi_turn_conversation(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [
            {"role": "user", "content": "My favorite number is 42."},
            {"role": "assistant", "content": "Got it, 42."},
            {"role": "user", "content": "What is my favorite number?"},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())


@pytest.mark.parametrize("param,value", [
    ("temperature", 0.0),
    ("temperature", 1.5),
    ("top_p", 0.5),
    ("max_tokens", 16),
    ("presence_penalty", 1.0),
    ("frequency_penalty", -1.0),
    ("n", 1),
    ("seed", 1234),
    ("stop", "STOP"),
    ("stop", ["STOP", "END"]),
    ("user", "conformance-test-user"),
])
def test_optional_sampling_params_accepted(openai_session, openai_base_url, openai_headers, openai_model, param, value):
    """Router should accept every documented optional param without erroring,
    even if it silently ignores ones it can't forward upstream."""
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
        param: value,
    })
    assert resp.status_code == 200, f"param {param}={value!r} -> {resp.status_code}: {resp.text}"
    validate_openai_chat_completion(resp.json())


def test_n_greater_than_one_returns_n_choices(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
        "n": 2,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    assert len(body["choices"]) == 2, f"requested n=2 but got {len(body['choices'])} choices"


def test_max_tokens_is_respected_via_finish_reason(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Write a long paragraph about the ocean."}],
        "max_tokens": 4,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    assert body["choices"][0]["finish_reason"] == "length", (
        f"expected finish_reason='length' when max_tokens is tiny, got {body['choices'][0]['finish_reason']!r}"
    )


def test_response_format_json_object(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Return a JSON object with a single key 'ok' set to true."}],
        "response_format": {"type": "json_object"},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    import json as _json
    content = body["choices"][0]["message"].get("content", "")
    _json.loads(content)  # must parse as valid JSON


def test_content_as_array_of_parts(openai_session, openai_base_url, openai_headers, openai_model):
    """Messages content may be a string OR an array of content parts."""
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello as content-part array"}]},
        ],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())


def test_finish_reason_stop_on_normal_completion(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Say 'done'."}],
        "max_tokens": 50,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_missing_required_field_messages_is_400(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {"model": openai_model})
    assert resp.status_code == 400, f"expected 400 for missing 'messages', got {resp.status_code}: {resp.text}"
    validate_openai_error(resp.json(), resp.status_code)


def test_missing_required_field_model_is_400(openai_session, openai_base_url, openai_headers):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert resp.status_code == 400, f"expected 400 for missing 'model', got {resp.status_code}: {resp.text}"
    validate_openai_error(resp.json(), resp.status_code)


def test_unknown_model_is_404_or_400(openai_session, openai_base_url, openai_headers):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": "definitely-not-a-real-model-xyz",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code in (400, 404), f"expected 400/404 for unknown model, got {resp.status_code}: {resp.text}"
    validate_openai_error(resp.json(), resp.status_code)


def test_empty_messages_array_is_400(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [],
    })
    assert resp.status_code == 400, f"expected 400 for empty messages array, got {resp.status_code}"
    validate_openai_error(resp.json(), resp.status_code)


def test_invalid_role_is_400(openai_session, openai_base_url, openai_headers, openai_model):
    resp = post_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "not-a-real-role", "content": "hi"}],
    })
    assert resp.status_code == 400, f"expected 400 for invalid role, got {resp.status_code}: {resp.text}"
    validate_openai_error(resp.json(), resp.status_code)


def test_content_type_json_required(openai_session, openai_base_url, openai_api_key, openai_model):
    resp = openai_session.post(
        f"{openai_base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai_api_key}"},  # no Content-Type
        data='{"model": "%s", "messages": [{"role":"user","content":"hi"}]}' % openai_model,
    )
    # Many frameworks are lenient here; router should not 500.
    assert resp.status_code < 500, f"server errored on missing Content-Type header: {resp.status_code}: {resp.text}"
