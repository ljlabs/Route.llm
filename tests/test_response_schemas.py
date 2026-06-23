"""Tests for SSE response format validation schemas."""

import json
import pytest
from core.translation.response_schemas import (
    validate_anthropic_sse_event,
    validate_anthropic_sse_line,
    validate_openai_sse_line,
    AnthropicMessageStartData,
    AnthropicContentBlockStartData,
    AnthropicContentBlockDeltaData,
    AnthropicContentBlockStopData,
    AnthropicMessageDeltaData,
    AnthropicMessageStopData,
    AnthropicPingData,
    OpenAIChatCompletionChunk,
)


# ---------------------------------------------------------------------------
# Anthropic schema unit tests
# ---------------------------------------------------------------------------

class TestAnthropicSchemas:
    def test_message_start_valid(self):
        data = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-opus",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 25, "output_tokens": 1},
            },
        }
        obj = AnthropicMessageStartData.model_validate(data)
        assert obj.type == "message_start"

    def test_message_start_missing_message(self):
        data = {"type": "message_start"}
        with pytest.raises(Exception):
            AnthropicMessageStartData.model_validate(data)

    def test_content_block_start_valid(self):
        data = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        obj = AnthropicContentBlockStartData.model_validate(data)
        assert obj.index == 0

    def test_content_block_start_tool_use(self):
        data = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "get_weather",
                "input": {},
            },
        }
        obj = AnthropicContentBlockStartData.model_validate(data)
        assert obj.content_block["type"] == "tool_use"

    def test_content_block_delta_text(self):
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        obj = AnthropicContentBlockDeltaData.model_validate(data)
        assert obj.delta["text"] == "Hello"

    def test_content_block_delta_tool_input(self):
        data = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"loc'},
        }
        obj = AnthropicContentBlockDeltaData.model_validate(data)
        assert obj.delta["type"] == "input_json_delta"

    def test_content_block_stop_valid(self):
        data = {"type": "content_block_stop", "index": 0}
        obj = AnthropicContentBlockStopData.model_validate(data)
        assert obj.index == 0

    def test_message_delta_valid(self):
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 15},
        }
        obj = AnthropicMessageDeltaData.model_validate(data)
        assert obj.delta["stop_reason"] == "end_turn"

    def test_message_stop_valid(self):
        data = {"type": "message_stop"}
        obj = AnthropicMessageStopData.model_validate(data)
        assert obj.type == "message_stop"

    def test_ping_valid(self):
        data = {"type": "ping"}
        obj = AnthropicPingData.model_validate(data)
        assert obj.type == "ping"


# ---------------------------------------------------------------------------
# Anthropic full SSE line validation
# ---------------------------------------------------------------------------

class TestAnthropicSSELineValidation:
    def test_valid_message_start_line(self):
        data = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-opus",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        line = "data: " + json.dumps(data)
        assert validate_anthropic_sse_line(line) is True

    def test_valid_content_block_delta_line(self):
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello world"},
        }
        line = "data: " + json.dumps(data)
        assert validate_anthropic_sse_line(line) is True

    def test_valid_tool_use_start_line(self):
        data = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "Read",
                "input": {},
            },
        }
        line = "data: " + json.dumps(data)
        assert validate_anthropic_sse_line(line) is True

    def test_done_line_passes(self):
        assert validate_anthropic_sse_line("data: [DONE]") is True

    def test_empty_data_line_passes(self):
        assert validate_anthropic_sse_line("data: ") is True

    def test_non_data_line_passes(self):
        assert validate_anthropic_sse_line("event: message_start") is True

    def test_unknown_event_type_warns(self):
        data = {"type": "unknown_future_event", "foo": "bar"}
        line = "data: " + json.dumps(data)
        assert validate_anthropic_sse_line(line) is False

    def test_malformed_json_warns(self):
        line = "data: {not valid json"
        assert validate_anthropic_sse_line(line) is False


# ---------------------------------------------------------------------------
# OpenAI schema unit tests
# ---------------------------------------------------------------------------

class TestOpenAISchemas:
    def test_chat_completion_chunk_valid(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        obj = OpenAIChatCompletionChunk.model_validate(data)
        assert obj.object == "chat.completion.chunk"
        assert obj.choices[0].delta == {"content": "Hello"}

    def test_chat_completion_chunk_finish(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        obj = OpenAIChatCompletionChunk.model_validate(data)
        assert obj.choices[0].finish_reason == "stop"

    def test_chat_completion_chunk_wrong_object(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [],
        }
        with pytest.raises(Exception):
            OpenAIChatCompletionChunk.model_validate(data)

    def test_chat_completion_chunk_tool_calls(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "Read", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        obj = OpenAIChatCompletionChunk.model_validate(data)
        assert len(obj.choices[0].delta["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# OpenAI full SSE line validation
# ---------------------------------------------------------------------------

class TestOpenAISSELineValidation:
    def test_valid_chat_chunk_line(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        }
        line = "data: " + json.dumps(data)
        assert validate_openai_sse_line(line) is True

    def test_done_line_passes(self):
        assert validate_openai_sse_line("data: [DONE]") is True

    def test_empty_data_line_passes(self):
        assert validate_openai_sse_line("data: ") is True

    def test_non_data_line_passes(self):
        assert validate_openai_sse_line("event: something") is True

    def test_wrong_object_type_warns(self):
        data = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [],
        }
        line = "data: " + json.dumps(data)
        assert validate_openai_sse_line(line) is False

    def test_malformed_json_warns(self):
        line = "data: {bad json}"
        assert validate_openai_sse_line(line) is False

    def test_missing_required_fields_warns(self):
        data = {"object": "chat.completion.chunk"}
        line = "data: " + json.dumps(data)
        assert validate_openai_sse_line(line) is False


# ---------------------------------------------------------------------------
# Database response_format setting
# ---------------------------------------------------------------------------

class TestResponseFormatSetting:
    def test_get_response_format_default(self):
        import tempfile, os
        import database as db_mod

        orig = db_mod.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_mod.DB_PATH = f.name
        try:
            db_mod.init_db()
            assert db_mod.get_response_format() == "anthropic"
        finally:
            db_mod.DB_PATH = orig
            os.unlink(f.name)

    def test_set_response_format_openai(self):
        import tempfile, os
        import database as db_mod

        orig = db_mod.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_mod.DB_PATH = f.name
        try:
            db_mod.init_db()
            db_mod.set_response_format("openai")
            assert db_mod.get_response_format() == "openai"
        finally:
            db_mod.DB_PATH = orig
            os.unlink(f.name)

    def test_set_response_format_invalid_raises(self):
        import tempfile, os
        import database as db_mod

        orig = db_mod.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_mod.DB_PATH = f.name
        try:
            db_mod.init_db()
            with pytest.raises(ValueError):
                db_mod.set_response_format("xml")
        finally:
            db_mod.DB_PATH = orig
            os.unlink(f.name)

    def test_response_format_roundtrip(self):
        import tempfile, os
        import database as db_mod

        orig = db_mod.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_mod.DB_PATH = f.name
        try:
            db_mod.init_db()
            db_mod.set_response_format("openai")
            assert db_mod.get_response_format() == "openai"
            db_mod.set_response_format("anthropic")
            assert db_mod.get_response_format() == "anthropic"
        finally:
            db_mod.DB_PATH = orig
            os.unlink(f.name)
