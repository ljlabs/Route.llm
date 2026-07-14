"""
Cross-cutting robustness tests that apply to both specs: malformed
JSON, wrong auth, wrong HTTP method, unsupported routes.

These matter for "works with any client" because clients frequently
hit these edge cases (retry storms, malformed proxies, wrong verbs from
misconfigured SDKs) well before they hit interesting parameter combos.
"""
import pytest

pytestmark = pytest.mark.errors


# ---------------- OpenAI-side ----------------

def test_openai_malformed_json_body_400(openai_session, openai_base_url, openai_headers):
    resp = openai_session.post(
        f"{openai_base_url}/v1/chat/completions",
        headers=openai_headers,
        data="{not valid json",
    )
    assert resp.status_code == 400, f"expected 400 for malformed JSON, got {resp.status_code}: {resp.text}"


def test_openai_wrong_api_key_401(openai_session, openai_base_url, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers={
        "Authorization": "Bearer clearly-invalid-key",
        "Content-Type": "application/json",
    }, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401, f"expected 401 for invalid API key, got {resp.status_code}: {resp.text}"


def test_openai_missing_auth_header_401(openai_session, openai_base_url, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers={
        "Content-Type": "application/json",
    }, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401, f"expected 401 with no Authorization header, got {resp.status_code}: {resp.text}"


def test_openai_wrong_http_method_405_or_404(openai_session, openai_base_url, openai_headers):
    resp = openai_session.get(f"{openai_base_url}/v1/chat/completions", headers=openai_headers)
    assert resp.status_code in (404, 405), f"expected 404/405 for GET on a POST-only route, got {resp.status_code}"


def test_openai_unsupported_route_404(openai_session, openai_base_url, openai_headers):
    resp = openai_session.get(f"{openai_base_url}/v1/definitely-not-a-real-route", headers=openai_headers)
    assert resp.status_code == 404


def test_openai_extra_unknown_fields_ignored_not_500(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
        "some_field_that_does_not_exist_in_the_spec": True,
    })
    assert resp.status_code < 500, f"unknown extra field caused a server error: {resp.status_code}: {resp.text}"


# ---------------- Anthropic-side ----------------

def test_anthropic_malformed_json_body_400(anthropic_session, anthropic_base_url, anthropic_headers):
    resp = anthropic_session.post(
        f"{anthropic_base_url}/v1/messages",
        headers=anthropic_headers,
        data="{not valid json",
    )
    assert resp.status_code == 400, f"expected 400 for malformed JSON, got {resp.status_code}: {resp.text}"


def test_anthropic_wrong_api_key_401(anthropic_session, anthropic_base_url, anthropic_version, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers={
        "x-api-key": "clearly-invalid-key",
        "anthropic-version": anthropic_version,
        "Content-Type": "application/json",
    }, json={
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 401, f"expected 401 for invalid API key, got {resp.status_code}: {resp.text}"


def test_anthropic_wrong_http_method_405_or_404(anthropic_session, anthropic_base_url, anthropic_headers):
    resp = anthropic_session.get(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers)
    assert resp.status_code in (404, 405), f"expected 404/405 for GET on a POST-only route, got {resp.status_code}"


def test_anthropic_unsupported_route_404(anthropic_session, anthropic_base_url, anthropic_headers):
    resp = anthropic_session.get(f"{anthropic_base_url}/v1/definitely-not-a-real-route", headers=anthropic_headers)
    assert resp.status_code == 404


def test_anthropic_extra_unknown_fields_ignored_not_500(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "some_field_that_does_not_exist_in_the_spec": True,
    })
    assert resp.status_code < 500, f"unknown extra field caused a server error: {resp.status_code}: {resp.text}"


def test_anthropic_max_tokens_must_be_positive_int(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": -5,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400, f"expected 400 for negative max_tokens, got {resp.status_code}: {resp.text}"
