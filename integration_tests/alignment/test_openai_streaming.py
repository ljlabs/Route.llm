"""
Conformance tests for streaming POST /v1/chat/completions (stream=true).
"""
import pytest
from .validators import validate_openai_stream_chunk
from .sse import iter_openai_sse

pytestmark = [pytest.mark.openai, pytest.mark.streaming]


def stream_chat(session, base_url, headers, payload):
    payload = {**payload, "stream": True}
    return session.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload, stream=True)


def test_stream_returns_sse_content_type(openai_session, openai_base_url, openai_headers, openai_model):
    resp = stream_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    })
    assert resp.status_code == 200, resp.text
    ctype = resp.headers.get("content-type", "")
    assert "text/event-stream" in ctype, f"expected text/event-stream, got {ctype!r}"


def test_stream_chunks_are_well_formed_and_terminate_with_done(openai_session, openai_base_url, openai_headers, openai_model):
    resp = stream_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    })
    assert resp.status_code == 200, resp.text
    chunks = list(iter_openai_sse(resp))
    assert len(chunks) >= 1, "expected at least one streamed chunk"
    for chunk in chunks:
        validate_openai_stream_chunk(chunk)

    # Reassemble content across chunks
    reassembled = "".join(
        choice["delta"].get("content", "")
        for chunk in chunks
        for choice in chunk["choices"]
    )
    assert len(reassembled) > 0, "no content accumulated across stream chunks"

    # last real chunk (before [DONE]) should carry a finish_reason
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0]["finish_reason"] is not None]
    assert finish_reasons, "no chunk carried a non-null finish_reason before the stream ended"


def test_stream_first_chunk_has_role_delta(openai_session, openai_base_url, openai_headers, openai_model):
    resp = stream_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200, resp.text
    chunks = list(iter_openai_sse(resp))
    assert chunks, "expected at least one chunk"
    first_delta = chunks[0]["choices"][0]["delta"]
    assert "role" in first_delta, "first streamed delta should announce role='assistant'"


def test_stream_with_usage_option(openai_session, openai_base_url, openai_headers, openai_model):
    """stream_options: {include_usage: true} should append a final chunk with usage."""
    resp = stream_chat(openai_session, openai_base_url, openai_headers, {
        "model": openai_model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream_options": {"include_usage": True},
    })
    assert resp.status_code == 200, resp.text
    chunks = list(iter_openai_sse(resp))
    has_usage = any(c.get("usage") is not None for c in chunks)
    assert has_usage, "expected a final chunk carrying 'usage' when stream_options.include_usage=true"
