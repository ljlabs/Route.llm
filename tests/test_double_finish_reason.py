"""
Tests for double finish_reason termination event fix.

ISSUE SUMMARY:
OpenRouter (and potentially other providers) send a trailing usage stats chunk
with finish_reason still set BEFORE the [DONE] sentinel. This causes the stream
translator to emit a complete termination sequence (content_block_stop → 
message_delta → message_stop) twice, confusing clients like Claude Code.

The fix uses a sent_stop guard flag to ensure termination events are emitted
exactly once, regardless of how many chunks have finish_reason != null.

Without this fix:
  [CHUNK N]   → finish_reason:"stop"   → emits: content_block_stop, message_delta, message_stop
  [CHUNK N+1] → finish_reason:"stop"   → emits: content_block_stop, message_delta, message_stop (DUPLICATE)
  [CHUNK N+2] → [DONE]

With this fix:
  [CHUNK N]   → finish_reason:"stop"   → emits: content_block_stop, message_delta, message_stop (sent_stop=True)
  [CHUNK N+1] → finish_reason:"stop"   → skipped (sent_stop guard prevents re-emit)
  [CHUNK N+2] → [DONE]
"""

import json
import pytest
from core.translation.stream_base import OpenAIToAnthropicStreamTranslator


class MockStreamResponse:
    """Mock streaming response for testing."""

    def __init__(self, chunks):
        self.chunks = chunks

    async def aiter_lines(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        pass


class TestDoubleFinishReasonFix:
    """Test that double finish_reason chunks don't emit duplicate termination events."""

    @pytest.mark.anyio
    async def test_openrouter_trailing_usage_chunk_does_not_duplicate_termination(self):
        """
        OpenRouter sends: normal chunks → finish_reason:stop → finish_reason:stop with usage → [DONE]
        
        Verify only ONE termination sequence is emitted.
        """
        chunks = [
            'data: {"id":"chat-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"id":"chat-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
            # First finish_reason chunk
            'data: {"id":"chat-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            # Trailing usage chunk ALSO with finish_reason (the problematic duplicate)
            'data: {"id":"chat-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":10,"completion_tokens":5}}]}',
            'data: [DONE]',
        ]
        
        response = MockStreamResponse(chunks)
        translator = OpenAIToAnthropicStreamTranslator()
        accumulated_blocks = []
        
        output_lines = []
        async for line in translator.translate_stream(response, {"name": "test", "model_name": "gpt-4"}, accumulated_blocks):
            output_lines.append(line)
        
        # Count termination events in output
        message_stop_count = sum(1 for line in output_lines if '"type": "message_stop"' in line)
        message_delta_count = sum(1 for line in output_lines if '"type": "message_delta"' in line)
        content_block_stop_count = sum(1 for line in output_lines if '"type": "content_block_stop"' in line)
        
        # Assert exactly one termination sequence
        assert message_stop_count == 1, f"Expected 1 message_stop event, got {message_stop_count}"
        assert message_delta_count == 1, f"Expected 1 message_delta event, got {message_delta_count}"
        assert content_block_stop_count == 1, f"Expected 1 content_block_stop event, got {content_block_stop_count}"
        
        # Assert content was accumulated only from first two chunks
        assert len(accumulated_blocks) == 1
        assert accumulated_blocks[0]["type"] == "text"
        assert accumulated_blocks[0]["text"] == "Hello world"

    @pytest.mark.anyio
    async def test_text_and_tool_emissions_guarded_by_sent_stop(self):
        """
        Verify that text and tool content from trailing chunks is not emitted
        once termination has started.
        """
        chunks = [
            'data: {"id":"msg-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            # Finish
            'data: {"id":"msg-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            # Trailing chunk with content (should be ignored)
            'data: {"id":"msg-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" ignored"},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]
        
        response = MockStreamResponse(chunks)
        translator = OpenAIToAnthropicStreamTranslator()
        accumulated_blocks = []
        
        output_lines = []
        async for line in translator.translate_stream(response, {"name": "test", "model_name": "gpt-4"}, accumulated_blocks):
            output_lines.append(line)
        
        # Only the first "Hello" should be accumulated, not " ignored"
        assert len(accumulated_blocks) == 1
        assert accumulated_blocks[0]["text"] == "Hello"
        
        # Count content_block_delta events (should be only 1 for "Hello")
        content_block_delta_count = sum(1 for line in output_lines if '"type": "content_block_delta"' in line and '"text_delta"' in line)
        assert content_block_delta_count == 1, f"Expected 1 content_block_delta for text, got {content_block_delta_count}"
