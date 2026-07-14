"""
Translation utilities for request/response format conversion.
These are used by the provider implementations.
"""

import json
import uuid
import time

SIGNATURE_SEPARATOR = "____ts____"


def _anthropic_image_to_openai(part: dict) -> dict:
    """Convert Anthropic image block to OpenAI image_url block."""
    source = part.get("source", {})
    media_type = source.get("media_type", "image/png")
    data = source.get("data", "")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{data}"
        }
    }


def _openai_image_url_to_anthropic(part: dict) -> dict:
    """Convert OpenAI image_url block to Anthropic image block."""
    url = part.get("image_url", {}).get("url", "")
    # Parse data URI: data:<media_type>;base64,<data>
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        # header = "data:image/jpeg;base64"
        media_type = header.split(":", 1)[1].split(";")[0] if ":" in header else "image/png"
    else:
        media_type = "image/png"
        data = url
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data
        }
    }


def sanitize_openai_payload(payload: dict, is_gemini: bool = False) -> dict:
    """
    Defensively strips Anthropic-specific or non-standard fields (like cache_control)
    and ensures content blocks are in a format strict APIs (like Mistral) expect.
    Preserves the original ordering of content blocks.
    """
    if "messages" not in payload:
        return payload

    for message in payload["messages"]:
        # Skip sanitization for tool role messages - preserve content as-is for image support
        if message.get("role") == "tool":
            message.pop("cache_control", None)
            continue
        
        content = message.get("content")
        
        # 1. Handle list content - preserve ordering but strip cache_control
        if isinstance(content, list):
            text_parts = []
            has_non_text = False
            
            for part in content:
                if isinstance(part, dict):
                    # Strip Anthropic-specific fields
                    part.pop("cache_control", None)
                    
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    else:
                        has_non_text = True
                elif isinstance(part, str):
                    text_parts.append(part)
            
            # If only text parts, flatten to string
            if text_parts and not has_non_text:
                message["content"] = "".join(text_parts)
            # If mixed or only non-text, keep as list with original ordering
            elif not (text_parts and not has_non_text):
                # Content stays as list with cache_control stripped above
                pass
        
        # 2. Strip any other top-level non-standard fields from message if they exist
        if isinstance(message, dict):
            message.pop("cache_control", None)
            
            # 3. Strip Gemini-specific extra_content from tool_calls if NOT a Gemini provider
            # Strict APIs like Mistral reject unknown fields in tool_calls
            if not is_gemini and "tool_calls" in message:
                tool_calls = message["tool_calls"]
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if isinstance(call, dict):
                            call.pop("extra_content", None)

    return payload


def anthropic_to_openai_request(anth_req: dict, target_model: str) -> dict:
    """Translates Anthropic /v1/messages request to OpenAI /v1/chat/completions request (including tools)"""
    messages = []
    
    # Add system prompt if present
    if "system" in anth_req and anth_req["system"]:
        system_content = anth_req["system"]
        # Defensive: Flatten system prompt to string if it's a list (Mistral/OpenAI strictness)
        if isinstance(system_content, list):
            text_parts = []
            for part in system_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            system_content = "".join(text_parts)
            
        messages.append({
            "role": "system",
            "content": system_content
        })
        
    for msg in anth_req.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        
        # Handle complex message content
        if isinstance(content, list):
            text_content = ""
            tool_calls = []
            image_blocks = []
            
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                
                if part_type == "text":
                    text_content += part.get("text", "")
                elif part_type == "image":
                    image_blocks.append(_anthropic_image_to_openai(part))
                elif part_type == "image_url":
                    # Already in OpenAI format — pass through as-is
                    image_blocks.append(part)
                elif part_type == "tool_use":
                    tool_id = part.get("id")
                    signature = None
                    if tool_id and SIGNATURE_SEPARATOR in tool_id:
                        parts = tool_id.split(SIGNATURE_SEPARATOR)
                        tool_id = parts[0]
                        signature = parts[1]
                    
                    tool_call = {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": part.get("name"),
                            "arguments": json.dumps(part.get("input", {}))
                        }
                    }
                    if signature:
                        tool_call["extra_content"] = {
                            "google": {
                                "thought_signature": signature
                            }
                        }
                    tool_calls.append(tool_call)
                elif part_type == "tool_result":
                    # Split tool results: text goes in the "tool" role message,
                    # images go in a subsequent "user" message per the OpenAI multi-turn pattern.
                    tool_result_content = part.get("content")
                    text_parts = []
                    image_parts = []

                    if isinstance(tool_result_content, list):
                        for sub_part in tool_result_content:
                            if isinstance(sub_part, dict):
                                if sub_part.get("type") == "text":
                                    text_parts.append(sub_part.get("text", ""))
                                elif sub_part.get("type") == "image":
                                    image_parts.append(_anthropic_image_to_openai(sub_part))
                    elif isinstance(tool_result_content, str):
                        text_parts.append(tool_result_content)

                    tool_use_id = part.get("tool_use_id")
                    if tool_use_id and SIGNATURE_SEPARATOR in tool_use_id:
                        tool_use_id = tool_use_id.split(SIGNATURE_SEPARATOR)[0]

                    if image_parts:
                        # Acknowledge the tool ran with a simple text confirmation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": json.dumps({"status": "success", "message": "Image loaded into context."})
                        })
                        # Append the image(s) as a separate user message
                        user_image_content = []
                        if text_parts:
                            user_image_content.append({"type": "text", "text": "".join(text_parts)})
                        else:
                            user_image_content.append({"type": "text", "text": "Here is the image you requested from the file path:"})
                        user_image_content.extend(image_parts)
                        messages.append({
                            "role": "user",
                            "content": user_image_content
                        })
                    else:
                        # No images — plain text tool result, standard tool role message
                        tool_text = "".join(text_parts) if text_parts else ""
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tool_text
                        })
            
            # If there's text, tool calls, or images for this turn, append the message
            if text_content or tool_calls or image_blocks:
                if image_blocks:
                    # Build content array with text + images
                    content_parts = []
                    if text_content:
                        content_parts.append({"type": "text", "text": text_content})
                    content_parts.extend(image_blocks)
                    openai_msg = {
                        "role": role,
                        "content": content_parts
                    }
                else:
                    openai_msg = {
                        "role": role,
                        "content": text_content
                    }
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                messages.append(openai_msg)
                
        else:
            # Simple string content
            messages.append({
                "role": role,
                "content": content
            })
            
    openai_req = {
        "model": target_model,
        "messages": messages,
    }
    
    # Translate Tools definition
    if "tools" in anth_req:
        openai_tools = []
        for tool in anth_req["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        if openai_tools:
            openai_req["tools"] = openai_tools
            
    # Translate parameters
    if "max_tokens" in anth_req:
        openai_req["max_tokens"] = anth_req["max_tokens"]
    if "temperature" in anth_req:
        openai_req["temperature"] = anth_req["temperature"]
    if "top_p" in anth_req:
        openai_req["top_p"] = anth_req["top_p"]
    if "stream" in anth_req:
        openai_req["stream"] = anth_req["stream"]
    # `top_k`, metadata and thinking are Anthropic-specific and are not
    # accepted by OpenAI-compatible upstreams. They are compatibility options:
    # accept them at the public boundary but do not forward unsupported fields.
    if "stop_sequences" in anth_req:
        openai_req["stop"] = anth_req["stop_sequences"]
    if "tool_choice" in anth_req:
        choice = anth_req["tool_choice"]
        if isinstance(choice, dict) and choice.get("type") == "tool":
            openai_req["tool_choice"] = {"type": "function", "function": {"name": choice.get("name", "")}}
        elif isinstance(choice, dict) and choice.get("type") == "none":
            openai_req["tool_choice"] = "none"
        else:
            openai_req["tool_choice"] = choice

    return openai_req


def openai_to_anthropic_request(openai_req: dict, target_model: str) -> dict:
    """Translates OpenAI /v1/chat/completions request to Anthropic /v1/messages request (including tools)"""
    messages = []
    system_prompt = None
    tool_results_buffer = []  # Buffer to accumulate consecutive tool results
    
    for msg in openai_req.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        
        if role == "system":
            system_prompt = content
        elif role == "tool":
            # Accumulate tool results into a buffer
            tool_result_content = []
            if content:
                if isinstance(content, list):
                    # Preserve mixed text and image content
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                tool_result_content.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "image_url":
                                tool_result_content.append(_openai_image_url_to_anthropic(part))
                elif isinstance(content, str):
                    tool_result_content.append({"type": "text", "text": content})
            
            tool_results_buffer.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": tool_result_content if tool_result_content else ""
            })
        else:
            # When we hit a non-tool message, flush the tool results buffer first
            if tool_results_buffer:
                messages.append({
                    "role": "user",
                    "content": tool_results_buffer
                })
                tool_results_buffer = []
            
            anth_content = []
            if content:
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                anth_content.append({
                                    "type": "text",
                                    "text": part.get("text", "")
                                })
                            elif part.get("type") == "image_url":
                                anth_content.append(_openai_image_url_to_anthropic(part))
                else:
                    anth_content.append({
                        "type": "text",
                        "text": content
                    })
            
            for call in tool_calls:
                call_func = call.get("function", {})
                try:
                    args = json.loads(call_func.get("arguments", "{}"))
                except Exception:
                    args = {}
                
                tool_id = call.get("id")
                # Try to restore signature if present in extra_content
                signature = call.get("extra_content", {}).get("google", {}).get("thought_signature")
                if signature:
                    tool_id = f"{tool_id}{SIGNATURE_SEPARATOR}{signature}"
                    
                anth_content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": call_func.get("name"),
                    "input": args
                })
                
            messages.append({
                "role": role,
                "content": anth_content if anth_content else ""
            })
    
    # Flush any remaining tool results at the end
    if tool_results_buffer:
        messages.append({
            "role": "user",
            "content": tool_results_buffer
        })
            
            
    anth_req = {
        "model": target_model,
        "messages": messages,
    }
    
    if system_prompt:
        anth_req["system"] = system_prompt
        
    # Translate Tools definition
    if "tools" in openai_req:
        anth_tools = []
        for tool in openai_req["tools"]:
            if tool.get("type") == "function":
                func = tool["function"]
                anth_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        if anth_tools:
            anth_req["tools"] = anth_tools
        
    if "max_tokens" in openai_req:
        anth_req["max_tokens"] = openai_req["max_tokens"]
    else:
        anth_req["max_tokens"] = 4096

    if "temperature" in openai_req:
        anth_req["temperature"] = openai_req["temperature"]
    if "top_p" in openai_req:
        anth_req["top_p"] = openai_req["top_p"]
    if "stream" in openai_req:
        anth_req["stream"] = openai_req["stream"]
    # OpenAI-only sampling and presentation controls are intentionally not
    # forwarded to an Anthropic upstream. They remain available to the router
    # for response normalization, while this payload stays provider-valid.
    if "stop" in openai_req:
        anth_req["stop_sequences"] = openai_req["stop"] if isinstance(openai_req["stop"], list) else [openai_req["stop"]]
    if "tool_choice" in openai_req:
        choice = openai_req["tool_choice"]
        if isinstance(choice, dict) and choice.get("type") == "function":
            anth_req["tool_choice"] = {"type": "tool", "name": (choice.get("function") or {}).get("name", "")}
        elif choice == "none":
            anth_req["tool_choice"] = {"type": "none"}
        else:
            anth_req["tool_choice"] = choice

    return anth_req


def openai_to_anthropic_response(openai_res: dict) -> dict:
    """Translates OpenAI non-streaming response to Anthropic response (including tool calls)"""
    choices = openai_res.get("choices", [])
    content_blocks = []
    stop_reason = "end_turn"
    
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        text_content = message.get("content", "")
        if text_content:
            content_blocks.append({
                "type": "text",
                "text": text_content
            })
            
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            stop_reason = "tool_use"
            for call in tool_calls:
                call_func = call.get("function", {})
                try:
                    args = json.loads(call_func.get("arguments", "{}"))
                except Exception:
                    args = {}
                
                tool_id = call.get("id", f"toolu_{uuid.uuid4().hex}")
                # Embed thought_signature if present
                signature = call.get("extra_content", {}).get("google", {}).get("thought_signature")
                if signature:
                    tool_id = f"{tool_id}{SIGNATURE_SEPARATOR}{signature}"
                    
                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": call_func.get("name"),
                    "input": args
                })
                
        finish_reason = choice.get("finish_reason")
        if finish_reason == "stop" and not tool_calls:
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
            
    return {
        "id": openai_res.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": openai_res.get("model", ""),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": openai_res.get("stop_sequence"),  # Pass through if available
        "usage": {
            "input_tokens": (openai_res.get("usage") or {}).get("prompt_tokens", 0),
            "output_tokens": (openai_res.get("usage") or {}).get("completion_tokens", 0)
        }
    }


def anthropic_to_openai_response(anth_res: dict) -> dict:
    """Translates Anthropic non-streaming response to OpenAI response (including tool calls)"""
    content = anth_res.get("content", [])
    text_content = ""
    tool_calls = []
    
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_content += part.get("text", "")
        elif part_type == "tool_use":
            tool_calls.append({
                "id": part.get("id"),
                "type": "function",
                "function": {
                    "name": part.get("name"),
                    "arguments": json.dumps(part.get("input", {}))
                }
            })
            
    stop_reason = "stop"
    anth_stop = anth_res.get("stop_reason")
    if anth_stop == "end_turn":
        stop_reason = "stop"
    elif anth_stop == "max_tokens":
        stop_reason = "length"
    elif anth_stop == "tool_use":
        stop_reason = "tool_calls"

    openai_choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": text_content if text_content else None
        },
        "finish_reason": stop_reason
    }
    if tool_calls:
        openai_choice["message"]["tool_calls"] = tool_calls

    # Build response with all available fields
    response = {
        "id": anth_res.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anth_res.get("model", ""),
        "choices": [openai_choice],
        "usage": {
            "prompt_tokens": anth_res.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": anth_res.get("usage", {}).get("output_tokens", 0),
            "total_tokens": anth_res.get("usage", {}).get("input_tokens", 0) + anth_res.get("usage", {}).get("output_tokens", 0)
        }
    }

    # Add system_fingerprint if available
    if "system_fingerprint" in anth_res:
        response["system_fingerprint"] = anth_res["system_fingerprint"]

    return response