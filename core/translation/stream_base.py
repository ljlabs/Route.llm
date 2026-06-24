"""
Stream Translator Base Classes and Implementations

Each stream translator handles conversion between Anthropic and OpenAI SSE streaming formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


class StreamTranslator(ABC):
    """Abstract base class for stream translators."""
    
    @abstractmethod
    async def translate_stream(
        self, 
        response, 
        provider_config: Dict[str, Any],
        accumulated_blocks: list
    ) -> Generator[str, None, None]:
        """
        Translate streaming response from one format to another.
        
        Args:
            response: The streaming response from the provider
            provider_config: Configuration info about the provider
            accumulated_blocks: List to accumulate response blocks
            
        Yields:
            Translated SSE lines
        """
        pass


class PassthroughStreamTranslator(StreamTranslator):
    """Pass-through translator when no translation is needed."""
    
    def __init__(self, validate_format: Optional[str] = None):
        self.validate_format = validate_format
        self.validation_warnings = []
        self.validation_checked = 0

    async def translate_stream(
        self, 
        response, 
        provider_config: Dict[str, Any],
        accumulated_blocks: list
    ) -> Generator[str, None, None]:
        """No translation needed - pass through the stream as-is."""
        from .response_schemas import validate_anthropic_sse_line, validate_openai_sse_line
        stream_logger = logging.getLogger("streaming")
        chunk_count = 0
        try:
            async for line in response.aiter_lines():
                chunk_count += 1
                stream_logger.debug(f"[LLM → ROUTER] Chunk {chunk_count} from {provider_config.get('name', 'Unknown')}: {line[:200]}...")
                yield line + "\n"
                
                # Validate output format if configured
                if self.validate_format and line:
                    if self.validate_format == "anthropic" and line.startswith("data:"):
                        self.validation_checked += 1
                        if not validate_anthropic_sse_line(line):
                            self.validation_warnings.append(line[:120])
                    elif self.validate_format == "openai" and line.startswith("data:"):
                        self.validation_checked += 1
                        if not validate_openai_sse_line(line):
                            self.validation_warnings.append(line[:120])
                
                # Still accumulate blocks for logging
                if line and line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    if data_content and data_content != "[DONE]":
                        try:
                            data = json.loads(data_content)
                            # Try to accumulate content
                            if provider_config.get("api_type") == "anthropic":
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if len(accumulated_blocks) <= 0:
                                        accumulated_blocks.append({"type": "text", "text": ""})
                                    accumulated_blocks[0]["text"] += text
                            else:
                                text = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if text:
                                    if len(accumulated_blocks) <= 0:
                                        accumulated_blocks.append({"type": "text", "text": ""})
                                    accumulated_blocks[0]["text"] += text
                        except Exception:
                            pass
            stream_logger.debug(f"[LLM → ROUTER] Stream completed after {chunk_count} chunks from {provider_config.get('name', 'Unknown')}")
        finally:
            await response.aclose()


class AnthropicToOpenAIStreamTranslator(StreamTranslator):
    """Translates OpenAI SSE stream to Anthropic SSE format."""
    
    def __init__(self, validate_format: Optional[str] = None):
        self.validate_format = validate_format
        self.validation_warnings = []
        self.validation_checked = 0
    
    async def translate_stream(
        self, 
        response, 
        provider_config: Dict[str, Any],
        accumulated_blocks: list
    ) -> Generator[str, None, None]:
        """Translate OpenAI streaming response to Anthropic format."""
        from .response_schemas import validate_anthropic_sse_line
        stream_logger = logging.getLogger("streaming")
        sent_start = False
        sent_stop = False
        msg_id = None  # Extract and reuse from first chunk
        tool_idx_map = {}
        next_anth_block_idx = 1
        llm_chunk_count = 0
        output_chunk_count = 0
        
        try:
            async for line in response.aiter_lines():
                llm_chunk_count += 1
                stream_logger.debug(f"[LLM → ROUTER] Raw chunk {llm_chunk_count} from {provider_config.get('name', 'Unknown')}: {line[:200]}...")
                if not line:
                    continue
                    
                if line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    if data_content == "[DONE]":
                        break
                    try:
                        data = json.loads(data_content)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        
                        # Extract message ID from first chunk and reuse throughout
                        if msg_id is None and data.get("id"):
                            msg_id = data.get("id")
                        
                        # Generate message start events once
                        if not sent_start:
                            # Use extracted ID or generate a fallback
                            if msg_id is None:
                                msg_id = f"msg_local_{os.urandom(8).hex()}"
                            model_name = data.get("model", provider_config.get("model_name", ""))
                            msg_start_data = {
                                "type": "message_start",
                                "message": {
                                    "id": msg_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model_name,
                                    "content": [],
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {"input_tokens": 0, "output_tokens": 0}
                                }
                            }
                            if self.validate_format == "anthropic":
                                self.validation_checked += 1
                                if not validate_anthropic_sse_line("data: " + json.dumps(msg_start_data)):
                                    self.validation_warnings.append("message_start")
                            output = f"event: message_start\ndata: " + json.dumps(msg_start_data) + "\n\n"
                            stream_logger.debug(f"[ROUTER → CLIENT] Output chunk: {output[:200]}...")
                            yield output
                            
                            # Start default text content block at index 0
                            cb_start_data = {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""}
                            }
                            if self.validate_format == "anthropic":
                                self.validation_checked += 1
                                if not validate_anthropic_sse_line("data: " + json.dumps(cb_start_data)):
                                    self.validation_warnings.append("content_block_start")
                            output = f"event: content_block_start\ndata: " + json.dumps(cb_start_data) + "\n\n"
                            stream_logger.debug(f"[ROUTER → CLIENT] Output chunk: {output[:200]}...")
                            yield output
                            sent_start = True
                        
                        # Yield standard text content chunk
                        text = delta.get("content", "")
                        if text and not sent_stop:
                            # Accumulate response block text
                            if len(accumulated_blocks) <= 0:
                                accumulated_blocks.append({"type": "text", "text": ""})
                            accumulated_blocks[0]["text"] += text
                            
                            cb_delta_data = {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text}
                            }
                            if self.validate_format == "anthropic":
                                self.validation_checked += 1
                                if not validate_anthropic_sse_line("data: " + json.dumps(cb_delta_data)):
                                    self.validation_warnings.append("content_block_delta")
                            output = f"event: content_block_delta\ndata: " + json.dumps(cb_delta_data) + "\n\n"
                            stream_logger.debug(f"[ROUTER → CLIENT] Output chunk: {output[:200]}...")
                            yield output
                            
                        # Yield tool calls chunks if present
                        tool_calls = delta.get("tool_calls", [])
                        for call in tool_calls:
                            if sent_stop:
                                break
                            call_idx = call.get("index", 0)
                            
                            # If new tool call index, send start event
                            if call_idx not in tool_idx_map:
                                anth_block_id = f"toolu_{os.urandom(8).hex()}"
                                tool_name = call.get("function", {}).get("name", "unknown_tool")
                                
                                signature = call.get("extra_content", {}).get("google", {}).get("thought_signature")
                                if signature:
                                    anth_block_id = f"{anth_block_id}____ts____{signature}"
                                    
                                tool_idx_map[call_idx] = {
                                    "id": anth_block_id,
                                    "name": tool_name,
                                    "arguments_accum": "",
                                    "anth_idx": next_anth_block_idx
                                }
                                
                                yield f"event: content_block_start\ndata: " + json.dumps({
                                    "type": "content_block_start",
                                    "index": next_anth_block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": anth_block_id,
                                        "name": tool_name,
                                        "input": {}
                                    }
                                }) + "\n\n"
                                next_anth_block_idx += 1
                            
                            # Yield argument json content deltas
                            arg_delta = call.get("function", {}).get("arguments", "")
                            if arg_delta:
                                tool_idx_map[call_idx]["arguments_accum"] += arg_delta
                                yield f"event: content_block_delta\ndata: " + json.dumps({
                                    "type": "content_block_delta",
                                    "index": tool_idx_map[call_idx]["anth_idx"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": arg_delta
                                    }
                                }) + "\n\n"
                            
                        # Handle stop reasons
                        if choice.get("finish_reason") is not None and not sent_stop:
                            sent_stop = True
                            finish_reason = choice.get("finish_reason")
                            stop_reason = "end_turn"
                            
                            # Stop text content block
                            yield f"event: content_block_stop\ndata: " + json.dumps({
                                "type": "content_block_stop",
                                "index": 0
                            }) + "\n\n"
                            
                            # Stop any active tool blocks
                            for c_idx, t_data in tool_idx_map.items():
                                yield f"event: content_block_stop\ndata: " + json.dumps({
                                    "type": "content_block_stop",
                                    "index": t_data["anth_idx"]
                                }) + "\n\n"
                                
                                # Try parsing accumulated json arguments
                                try:
                                    parsed_args = json.loads(t_data["arguments_accum"])
                                except Exception:
                                    parsed_args = t_data["arguments_accum"]
                                accumulated_blocks.append({
                                    "type": "tool_use",
                                    "id": t_data["id"],
                                    "name": t_data["name"],
                                    "input": parsed_args
                                })
                            
                            if finish_reason == "tool_calls" or tool_idx_map:
                                stop_reason = "tool_use"
                            elif finish_reason == "length":
                                stop_reason = "max_tokens"
                            
                            yield f"event: message_delta\ndata: " + json.dumps({
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                "usage": {"output_tokens": 0}
                            }) + "\n\n"
                            yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
                    except Exception:
                        pass
                        
        except Exception as stream_err:
            logger.error(f"Stream translation error: {stream_err}")
            yield "event: error\ndata: " + json.dumps({
                "type": "error", 
                "error": {"type": "api_error", "message": str(stream_err)}
            }) + "\n\n"
        finally:
            await response.aclose()


class OpenAIToAnthropicStreamTranslator(StreamTranslator):
    """Translates Anthropic SSE stream to OpenAI format."""
    
    async def translate_stream(
        self, 
        response, 
        provider_config: Dict[str, Any],
        accumulated_blocks: list
    ) -> Generator[str, None, None]:
        """Translate Anthropic streaming response to OpenAI format."""
        sent_start = False
        msg_id = None  # Extract and reuse from message_start event
        
        try:
            async for line in response.aiter_lines():
                if not line:
                    continue
                    
                if line.startswith("data:"):
                    data_content = line.replace("data:", "").strip()
                    try:
                        data = json.loads(data_content)
                        event_type = data.get("type")
                        
                        if event_type == "message_start":
                            # Extract ID from message_start event
                            msg_id = data.get("message", {}).get("id")
                            # Generate fallback ID if provider doesn't provide one
                            if not msg_id:
                                msg_id = f"chatcmpl-{os.urandom(8).hex()}"
                            model_name = data.get("message", {}).get("model", provider_config.get("model_name", ""))
                            yield "data: " + json.dumps({
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": int(data.get("message", {}).get("created", 0)) if "created" in data.get("message", {}) else int(__import__('time').time()),
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }]
                            }) + "\n\n"
                            sent_start = True
                        elif event_type == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                if len(accumulated_blocks) <= 0:
                                    accumulated_blocks.append({"type": "text", "text": ""})
                                accumulated_blocks[0]["text"] += text
                                # Ensure msg_id is set before sending delta
                                if msg_id is None:
                                    msg_id = f"chatcmpl-{os.urandom(8).hex()}"
                                yield "data: " + json.dumps({
                                    "id": msg_id,  # REUSE the ID from message_start
                                    "object": "chat.completion.chunk",
                                    "created": int(__import__('time').time()),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None
                                    }]
                                }) + "\n\n"
                        elif event_type == "message_delta":
                            anth_stop = data.get("delta", {}).get("stop_reason")
                            stop_reason = "stop"
                            if anth_stop == "end_turn":
                                stop_reason = "stop"
                            elif anth_stop == "max_tokens":
                                stop_reason = "length"
                            
                            # Ensure msg_id is set before sending delta
                            if msg_id is None:
                                msg_id = f"chatcmpl-{os.urandom(8).hex()}"
                            yield "data: " + json.dumps({
                                "id": msg_id,  # REUSE the ID from message_start
                                "object": "chat.completion.chunk",
                                "created": int(__import__('time').time()),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": stop_reason
                                }]
                            }) + "\n\n"
                            yield "data: [DONE]\n\n"
                    except Exception:
                        pass
                        
        except Exception as stream_err:
            logger.error(f"Stream translation error: {stream_err}")
            yield "data: " + json.dumps({
                "choices": [{
                    "delta": {"content": f"\n[Stream Error: {str(stream_err)}]"},
                    "finish_reason": "error"
                }]
            }) + "\n\n"
        finally:
            await response.aclose()