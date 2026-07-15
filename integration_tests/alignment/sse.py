"""
Minimal Server-Sent-Events parser built on top of requests' streaming
iterator. Avoids adding a dependency on sseclient-py.

OpenAI-style streams: every event is a `data: {...}` line, terminated
by `data: [DONE]`. No explicit `event:` line.

Anthropic-style streams: paired `event: <type>` and `data: {...}` lines
per event, no terminal sentinel (stream just closes after message_stop).
"""
import json


def iter_openai_sse(response):
    """Yield parsed JSON dicts from an OpenAI-style SSE response, one per event.
    Stops before yielding the terminal [DONE] sentinel."""
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


def iter_anthropic_sse(response):
    """Yield (event_type, parsed_json_dict) tuples from an Anthropic-style
    SSE response."""
    event_type = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        if raw_line == "":
            event_type = None
            continue
        if raw_line.startswith("event:"):
            event_type = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            payload = raw_line[len("data:"):].strip()
            data = json.loads(payload)
            yield event_type, data
