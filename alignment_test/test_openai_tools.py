"""
Conformance tests for OpenAI-style tool / function calling.
"""
import json
import pytest
from validators import validate_openai_chat_completion
from sse import iter_openai_sse

pytestmark = [pytest.mark.openai, pytest.mark.tools]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


def test_tool_call_triggered_and_well_formed(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
        "tools": [WEATHER_TOOL],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)

    choice = body["choices"][0]
    if choice["finish_reason"] == "tool_calls":
        tool_calls = choice["message"]["tool_calls"]
        assert tool_calls, "finish_reason=tool_calls but no tool_calls present"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert "location" in args, "expected model to fill 'location' argument"
    else:
        pytest.skip("model chose not to call the tool for this prompt; router still returned a valid shape")


def test_tool_choice_forced_specific_function(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", (
        f"tool_choice forced a specific function; expected finish_reason='tool_calls', got {choice['finish_reason']!r}"
    )
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_tool_choice_none_suppresses_calls(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "none",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_chat_completion(body)
    assert body["choices"][0]["finish_reason"] != "tool_calls", "tool_choice='none' should suppress tool calls"


def test_tool_result_round_trip(openai_session, openai_base_url, openai_headers, openai_model):
    """Full round trip: user asks -> assistant calls tool -> we supply a
    'tool' role message with the result -> model produces a final answer."""
    first = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })
    assert first.status_code == 200, first.text
    first_body = first.json()
    tool_call = first_body["choices"][0]["message"]["tool_calls"][0]

    second = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [
            {"role": "user", "content": "What's the weather in Paris? Use the tool."},
            first_body["choices"][0]["message"],
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps({"temperature": 21, "unit": "celsius", "condition": "sunny"}),
            },
        ],
        "tools": [WEATHER_TOOL],
    })
    assert second.status_code == 200, second.text
    validate_openai_chat_completion(second.json())


def test_streaming_tool_call_deltas_assemble(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": True,
    }, stream=True)
    assert resp.status_code == 200, resp.text
    chunks = list(iter_openai_sse(resp))
    assert chunks, "expected streamed chunks"

    name_seen = False
    args_buffer = ""
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]
        for tc in delta.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if fn.get("name"):
                name_seen = True
            args_buffer += fn.get("arguments", "") or ""

    assert name_seen, "expected a tool_calls delta announcing the function name at some point in the stream"
    json.loads(args_buffer)  # accumulated arguments must form valid JSON
