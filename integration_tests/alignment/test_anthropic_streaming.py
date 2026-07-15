"""
Conformance tests for streaming POST /v1/messages (stream=true).

Anthropic's stream is an ordered event sequence:
  message_start
  content_block_start   (index 0)
  content_block_delta*  (index 0)
  content_block_stop    (index 0)
  ... repeated per content block ...
  message_delta         (carries stop_reason + cumulative usage)
  message_stop
"""
import pytest
from .validators import validate_anthropic_sse_event
from .sse import iter_anthropic_sse

pytestmark = [pytest.mark.anthropic, pytest.mark.streaming]


def stream_messages(session, base_url, headers, payload):
    payload = {**payload, "stream": True}
    return session.post(f"{base_url}/v1/messages", headers=headers, json=payload, stream=True)


def test_stream_content_type(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = stream_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    })
    assert resp.status_code == 200, resp.text
    ctype = resp.headers.get("content-type", "")
    assert "text/event-stream" in ctype, f"expected text/event-stream, got {ctype!r}"


def test_stream_event_sequence_well_formed(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = stream_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    })
    assert resp.status_code == 200, resp.text
    events = list(iter_anthropic_sse(resp))
    assert events, "expected at least one SSE event"

    for event_type, data in events:
        validate_anthropic_sse_event(event_type, data)

    types_in_order = [e for e, _ in events]
    assert types_in_order[0] == "message_start", f"first event should be message_start, got {types_in_order[0]!r}"
    assert "message_stop" in types_in_order, "stream must end with message_stop"
    assert types_in_order[-1] == "message_stop", f"last event should be message_stop, got {types_in_order[-1]!r}"

    assert "content_block_start" in types_in_order
    assert "content_block_delta" in types_in_order
    assert "content_block_stop" in types_in_order
    assert "message_delta" in types_in_order


def test_stream_text_reassembles(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = stream_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello."}],
    })
    assert resp.status_code == 200, resp.text
    events = list(iter_anthropic_sse(resp))

    text = ""
    for event_type, data in events:
        if event_type == "content_block_delta" and data["delta"].get("type") == "text_delta":
            text += data["delta"]["text"]
    assert len(text) > 0, "expected accumulated text from content_block_delta events"


def test_stream_message_delta_carries_stop_reason_and_usage(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = stream_messages(anthropic_session, anthropic_base_url, anthropic_headers, {
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello."}],
    })
    assert resp.status_code == 200, resp.text
    events = list(iter_anthropic_sse(resp))
    message_delta_events = [data for etype, data in events if etype == "message_delta"]
    assert message_delta_events, "expected a message_delta event"
    last = message_delta_events[-1]
    assert "stop_reason" in last["delta"], "message_delta.delta should carry stop_reason"
    assert "usage" in last, "message_delta should carry cumulative usage"
    assert "output_tokens" in last["usage"]
