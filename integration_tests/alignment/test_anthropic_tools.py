"""
Conformance tests for Anthropic-style tool use.
"""
import pytest
from .validators import validate_anthropic_message
from .sse import iter_anthropic_sse

pytestmark = [pytest.mark.anthropic, pytest.mark.tools]

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}


def test_tool_use_triggered_and_well_formed(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
        "tools": [WEATHER_TOOL],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_anthropic_message(body)

    if body["stop_reason"] == "tool_use":
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert tool_blocks, "stop_reason=tool_use but no tool_use content block present"
        assert "location" in tool_blocks[0]["input"], "expected 'location' filled in tool input"
    else:
        pytest.skip("model chose not to call the tool for this prompt; router still returned a valid shape")


def test_tool_choice_forced_tool(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_anthropic_message(body)
    assert body["stop_reason"] == "tool_use", (
        f"tool_choice forced a specific tool; expected stop_reason='tool_use', got {body['stop_reason']!r}"
    )
    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert tool_blocks and tool_blocks[0]["name"] == "get_weather"


def test_tool_choice_none_type(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "none"},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_anthropic_message(body)
    assert body["stop_reason"] != "tool_use", "tool_choice type='none' should suppress tool use"


def test_tool_result_round_trip(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    first = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    })
    assert first.status_code == 200, first.text
    first_body = first.json()
    tool_block = next(b for b in first_body["content"] if b["type"] == "tool_use")

    second = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris? Use the tool."},
            {"role": "assistant", "content": first_body["content"]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block["id"],
                "content": "21 degrees celsius and sunny",
            }]},
        ],
        "tools": [WEATHER_TOOL],
    })
    assert second.status_code == 200, second.text
    validate_anthropic_message(second.json())


def test_streaming_tool_use_input_json_deltas_assemble(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "stream": True,
    }, stream=True)
    assert resp.status_code == 200, resp.text
    events = list(iter_anthropic_sse(resp))

    import json as _json
    input_json_buffer = ""
    saw_tool_use_start = False
    for event_type, data in events:
        if event_type == "content_block_start" and data["content_block"].get("type") == "tool_use":
            saw_tool_use_start = True
        if event_type == "content_block_delta" and data["delta"].get("type") == "input_json_delta":
            input_json_buffer += data["delta"].get("partial_json", "")

    assert saw_tool_use_start, "expected a content_block_start with type='tool_use'"
    if input_json_buffer:
        _json.loads(input_json_buffer)  # must assemble into valid JSON
