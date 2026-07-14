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


def _without_null_values(value: Any) -> Any:
    """Recursively remove null values before emitting an SSE JSON payload."""
    if isinstance(value, dict):
        return {
            key: _without_null_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_null_values(item) for item in value if item is not None]
    return value


def _sanitize_sse_chunk(chunk: str) -> str:
    """Remove null JSON values from each data line while preserving SSE framing."""
    sanitized_lines = []
    for line in chunk.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body):]
        if not line_body.startswith("data:"):
            sanitized_lines.append(line)
            continue

        data_content = line_body.removeprefix("data:").strip()
        if not data_content or data_content == "[DONE]":
            sanitized_lines.append(line)
            continue

        try:
            data = json.loads(data_content)
        except json.JSONDecodeError:
            sanitized_lines.append(line)
            continue

        sanitized_data = _without_null_values(data)
        if sanitized_data is None:
            sanitized_data = {}
        compact_json = ": " not in data_content and ", " not in data_content
        json_kwargs = {"separators": (",", ":")} if compact_json else {}
        sanitized_lines.append(
            "data: " + json.dumps(sanitized_data, **json_kwargs) + line_ending
        )
    return "".join(sanitized_lines)


def _as_dict(value: Any) -> Dict[str, Any]:
    """Treat an explicitly null or malformed structural value as an empty object."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Treat an explicitly null or malformed structural value as an empty array."""
    return value if isinstance(value, list) else []


def _sanitize_stream_translation(translator_class):
    """Apply outbound null removal to every response produced by a translator."""
    original_translate_stream = translator_class.translate_stream

    async def translate_stream(self, *args, **kwargs):
        async for chunk in original_translate_stream(self, *args, **kwargs):
            yield _sanitize_sse_chunk(chunk)

    translator_class.translate_stream = translate_stream
    return translator_class


async def _iter_openai_sse_lines(response):
    """Yield provider SSE lines, adapting buffered OpenAI JSON completions when needed."""
    response_headers = getattr(response, "headers", {})
    content_type = str(response_headers.get("content-type", "")).lower()
    if "application/json" not in content_type:
        async for line in response.aiter_lines():
            yield line
        return

    raw_body = response.content if getattr(response, "is_stream_consumed", False) else await response.aread()
    if isinstance(raw_body, bytes):
        raw_body = raw_body.decode("utf-8", errors="replace")

    try:
        completion = _as_dict(json.loads(raw_body))
    except json.JSONDecodeError as exc:
        raise ValueError("Upstream returned invalid JSON for a streaming request") from exc

    choices = _as_list(completion.get("choices"))
    if not choices:
        raise ValueError("Upstream JSON response has no completion choices")

    choice = _as_dict(choices[0])
    message = _as_dict(choice.get("message"))
    choice_index = choice.get("index") or 0
    base_chunk = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created"),
        "model": completion.get("model"),
    }

    yield "data: " + json.dumps({
        **base_chunk,
        "choices": [{
            "index": choice_index,
            "delta": {"role": message.get("role") or "assistant"},
            "finish_reason": None,
        }],
    })

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    if content:
        yield "data: " + json.dumps({
            **base_chunk,
            "choices": [{
                "index": choice_index,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        })

    tool_calls = []
    for tool_index, raw_call in enumerate(_as_list(message.get("tool_calls"))):
        call = _as_dict(raw_call)
        if call:
            tool_calls.append({**call, "index": call.get("index") or tool_index})
    if tool_calls:
        yield "data: " + json.dumps({
            **base_chunk,
            "choices": [{
                "index": choice_index,
                "delta": {"tool_calls": tool_calls},
                "finish_reason": None,
            }],
        })

    yield "data: " + json.dumps({
        **base_chunk,
        "choices": [{
            "index": choice_index,
            "delta": {},
            "finish_reason": choice.get("finish_reason") or "stop",
        }],
    })
    yield "data: [DONE]"




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




@_sanitize_stream_translation
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
        response_content_type = str(
            getattr(response, "headers", {}).get("content-type", "")
        ).lower()
        upstream_lines = (
            _iter_openai_sse_lines(response)
            if "application/json" in response_content_type
            else response.aiter_lines()
        )
        line_suffix = "\n\n" if "application/json" in response_content_type else "\n"


        try:


            async for line in upstream_lines:


                chunk_count += 1


                stream_logger.debug(f"[LLM → ROUTER] Chunk {chunk_count} from {provider_config.get('name', 'Unknown')}: {line[:200]}...")


                yield line + line_suffix
                


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



@_sanitize_stream_translation
class OpenAIToAnthropicStreamTranslator(StreamTranslator):


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


        text_block_started = False  # Defer text block start until actual text arrives


        msg_id = None  # Extract and reuse from first chunk


        tool_idx_map = {}


        next_anth_block_idx = 1


        llm_chunk_count = 0


        output_chunk_count = 0
        


        try:


            async for line in _iter_openai_sse_lines(response):


                llm_chunk_count += 1


                stream_logger.debug(f"[LLM → ROUTER] Raw chunk {llm_chunk_count} from {provider_config.get('name', 'Unknown')}: {line[:200]}...")


                if not line:


                    continue
                    


                if line.startswith("data:"):


                    data_content = line.replace("data:", "").strip()


                    if data_content == "[DONE]":


                        break


                    try:


                        data = _as_dict(json.loads(data_content))


                        choices = _as_list(data.get("choices"))


                        if not choices:


                            continue


                        choice = _as_dict(choices[0])


                        delta = _as_dict(choice.get("delta"))
                        


                        # Extract message ID from first chunk and reuse throughout


                        if msg_id is None and data.get("id"):


                            msg_id = data.get("id")
                        


                        # Generate message start events once


                        if not sent_start:


                            # Use extracted ID or generate a fallback


                            if msg_id is None:


                                msg_id = f"msg_local_{os.urandom(8).hex()}"


                            model_name = data.get("model") or provider_config.get("model_name", "")


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
                            

                            # Defer text content_block_start until actual text arrives


                            sent_start = True
                        


                        # Yield standard text content chunk


                        text = delta.get("content", "")


                        if text and not sent_stop:
                            # Defer emit text content_block_start until we have actual text
                            if not text_block_started:
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
                                text_block_started = True
                            
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


                        tool_calls = _as_list(delta.get("tool_calls"))


                        for raw_call in tool_calls:


                            call = _as_dict(raw_call)
                            if not call:
                                continue


                            if sent_stop:


                                break


                            function = _as_dict(call.get("function"))
                            extra_content = _as_dict(call.get("extra_content"))
                            google_content = _as_dict(extra_content.get("google"))
                            call_idx = call.get("index") or 0
                            


                            # If new tool call index, send start event


                            if call_idx not in tool_idx_map:


                                anth_block_id = f"toolu_{os.urandom(8).hex()}"


                                tool_name = function.get("name") or "unknown_tool"
                                


                                signature = google_content.get("thought_signature")


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


                            arg_delta = function.get("arguments") or ""


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
                            

                            # Stop text content block (only if it was started)
                            if text_block_started:
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


                    except json.JSONDecodeError as e:
                        logger.warning(f"Malformed SSE chunk (JSON parse error): {e} — raw: {data_content[:200]}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error translating stream chunk: {e}", exc_info=True)
                        continue
                        


        except Exception as stream_err:


            logger.error(f"Stream translation error: {stream_err}")


            yield "event: error\ndata: " + json.dumps({


                "type": "error", 


                "error": {"type": "api_error", "message": str(stream_err)}


            }) + "\n\n"


        finally:


            await response.aclose()



@_sanitize_stream_translation
class AnthropicToOpenAIStreamTranslator(StreamTranslator):


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
        tool_blocks = {}  # Maps Anthropic block index → {id, name, openai_index}
        next_tool_index = 0
        


        try:


            async for line in response.aiter_lines():


                if not line:


                    continue
                    


                if line.startswith("data:"):


                    data_content = line.replace("data:", "").strip()


                    try:


                        data = _as_dict(json.loads(data_content))


                        event_type = data.get("type")
                        


                        if event_type == "message_start":


                            # Extract ID from message_start event


                            message = _as_dict(data.get("message"))
                            msg_id = message.get("id")


                            # Generate fallback ID if provider doesn't provide one


                            if not msg_id:


                                msg_id = f"chatcmpl-{os.urandom(8).hex()}"


                            model_name = message.get("model") or provider_config.get("model_name", "")


                            yield "data: " + json.dumps({


                                "id": msg_id,


                                "object": "chat.completion.chunk",


                                "created": int(message.get("created") or __import__('time').time()),


                                "model": model_name,


                                "choices": [{


                                    "index": 0,


                                    "delta": {"role": "assistant"},


                                    "finish_reason": None


                                }]


                            }) + "\n\n"


                            sent_start = True


                        elif event_type == "content_block_start":
                            block = _as_dict(data.get("content_block"))
                            block_index = data.get("index")
                            if block.get("type") == "tool_use":
                                tool_id = block.get("id") or ""
                                tool_name = block.get("name") or ""
                                tool_blocks[block_index] = {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "openai_index": next_tool_index
                                }
                                # Emit tool_calls start delta
                                yield "data: " + json.dumps({
                                    "id": msg_id or f"chatcmpl-{os.urandom(8).hex()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(__import__('time').time()),
                                    "model": provider_config.get("model_name", ""),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": next_tool_index,
                                                "id": tool_id,
                                                "type": "function",
                                                "function": {"name": tool_name, "arguments": ""}
                                            }]
                                        },
                                        "finish_reason": None
                                    }]
                                }) + "\n\n"
                                next_tool_index += 1

                        elif event_type == "content_block_delta":
                            delta = _as_dict(data.get("delta"))
                            delta_type = delta.get("type")
                            block_index = data.get("index")
                            
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
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
                            elif delta_type == "thinking_delta":
                                # Pass through thinking blocks as content
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    yield "data: " + json.dumps({
                                        "id": msg_id or f"chatcmpl-{os.urandom(8).hex()}",
                                        "object": "chat.completion.chunk",
                                        "created": int(__import__('time').time()),
                                        "model": provider_config.get("model_name", ""),
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": thinking_text},
                                            "finish_reason": None
                                        }]
                                    }) + "\n\n"
                            elif delta_type == "input_json_delta" and block_index in tool_blocks:
                                partial_json = delta.get("partial_json") or ""
                                tool_info = tool_blocks[block_index]
                                yield "data: " + json.dumps({
                                    "id": msg_id or f"chatcmpl-{os.urandom(8).hex()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(__import__('time').time()),
                                    "model": provider_config.get("model_name", ""),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": tool_info["openai_index"],
                                                "function": {"arguments": partial_json}
                                            }]
                                        },
                                        "finish_reason": None
                                    }]
                                }) + "\n\n"


                        elif event_type == "message_delta":
                            anth_stop = _as_dict(data.get("delta")).get("stop_reason")
                            stop_reason = "stop"
                            if anth_stop == "end_turn":
                                stop_reason = "stop"
                            elif anth_stop == "max_tokens":
                                stop_reason = "length"
                            elif anth_stop == "tool_use":
                                stop_reason = "tool_calls"
                            


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


                    except json.JSONDecodeError as e:
                        logger.warning(f"Malformed SSE chunk (JSON parse error): {e} — raw: {data_content[:200]}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error translating stream chunk: {e}", exc_info=True)
                        continue
                        


        except Exception as stream_err:
            logger.error(f"Stream translation error: {stream_err}", exc_info=True)
            # Emit clean termination instead of injecting error as content
            yield "data: " + json.dumps({
                "id": msg_id or f"chatcmpl-{os.urandom(8).hex()}",
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": provider_config.get("model_name", ""),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }) + "\n\n"
            yield "data: [DONE]\n\n"


        finally:


            await response.aclose()