"""
Structural validators for OpenAI and Anthropic response bodies.

These are intentionally hand-rolled (not full JSON Schema) so failures
produce a specific, readable assertion message pointing at exactly
which field is missing or wrong-typed, rather than a generic schema
validator dump.
"""


def require(condition, message):
    assert condition, message


def require_keys(obj, keys, ctx=""):
    for k in keys:
        require(k in obj, f"{ctx}missing required key '{k}' in {obj!r}")


def require_type(obj, key, expected_type, ctx=""):
    require(key in obj, f"{ctx}missing key '{key}'")
    require(
        isinstance(obj[key], expected_type),
        f"{ctx}key '{key}' expected type {expected_type}, got {type(obj[key])} ({obj[key]!r})",
    )


# ---------------- OpenAI ----------------

def validate_openai_chat_completion(body):
    ctx = "[openai chat.completion] "
    require_keys(body, ["id", "object", "created", "model", "choices"], ctx)
    require(body["object"] == "chat.completion", f"{ctx}object field must equal 'chat.completion', got {body['object']!r}")
    require_type(body, "created", int, ctx)
    require_type(body, "choices", list, ctx)
    require(len(body["choices"]) >= 1, f"{ctx}choices must be non-empty")

    for i, choice in enumerate(body["choices"]):
        cctx = f"{ctx}choices[{i}] "
        require_keys(choice, ["index", "message", "finish_reason"], cctx)
        msg = choice["message"]
        require_keys(msg, ["role"], cctx + "message ")
        require(msg["role"] == "assistant", f"{cctx}message.role must be 'assistant', got {msg['role']!r}")
        require(
            "content" in msg or "tool_calls" in msg,
            f"{cctx}message must contain 'content' and/or 'tool_calls'",
        )
        require(
            choice["finish_reason"] in (None, "stop", "length", "tool_calls", "content_filter", "function_call"),
            f"{cctx}unexpected finish_reason {choice['finish_reason']!r}",
        )
        if msg.get("tool_calls"):
            for j, tc in enumerate(msg["tool_calls"]):
                tctx = f"{cctx}tool_calls[{j}] "
                require_keys(tc, ["id", "type", "function"], tctx)
                require(tc["type"] == "function", f"{tctx}type must be 'function'")
                require_keys(tc["function"], ["name", "arguments"], tctx + "function ")
                require(isinstance(tc["function"]["arguments"], str), f"{tctx}function.arguments must be a JSON string")

    if "usage" in body:
        u = body["usage"]
        require_keys(u, ["prompt_tokens", "completion_tokens", "total_tokens"], ctx + "usage ")
        require(
            u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
            f"{ctx}usage.total_tokens should equal prompt_tokens + completion_tokens",
        )


def validate_openai_stream_chunk(chunk):
    ctx = "[openai chat.completion.chunk] "
    require_keys(chunk, ["id", "object", "created", "model", "choices"], ctx)
    require(
        chunk["object"] == "chat.completion.chunk",
        f"{ctx}object field must equal 'chat.completion.chunk', got {chunk['object']!r}",
    )
    require_type(chunk, "choices", list, ctx)
    for i, choice in enumerate(chunk["choices"]):
        cctx = f"{ctx}choices[{i}] "
        require_keys(choice, ["index", "delta"], cctx)
        require("finish_reason" in choice, f"{cctx}must include 'finish_reason' key (may be null)")


def validate_openai_error(body, status_code):
    ctx = f"[openai error, status {status_code}] "
    require_keys(body, ["error"], ctx)
    err = body["error"]
    require_keys(err, ["message", "type"], ctx + "error ")
    require(isinstance(err["message"], str) and len(err["message"]) > 0, f"{ctx}error.message must be non-empty string")


def validate_openai_models_list(body):
    ctx = "[openai models list] "
    require_keys(body, ["object", "data"], ctx)
    require(body["object"] == "list", f"{ctx}object must equal 'list'")
    require_type(body, "data", list, ctx)
    for i, m in enumerate(body["data"]):
        mctx = f"{ctx}data[{i}] "
        require_keys(m, ["id", "object"], mctx)
        require(m["object"] == "model", f"{mctx}object must equal 'model'")


# ---------------- Anthropic ----------------

def validate_anthropic_message(body):
    ctx = "[anthropic message] "
    require_keys(body, ["id", "type", "role", "content", "model", "stop_reason", "usage"], ctx)
    require(body["type"] == "message", f"{ctx}type must equal 'message', got {body['type']!r}")
    require(body["role"] == "assistant", f"{ctx}role must equal 'assistant'")
    require_type(body, "content", list, ctx)
    require(len(body["content"]) >= 1, f"{ctx}content must be non-empty")

    for i, block in enumerate(body["content"]):
        bctx = f"{ctx}content[{i}] "
        require_keys(block, ["type"], bctx)
        if block["type"] == "text":
            require_keys(block, ["text"], bctx)
            require(isinstance(block["text"], str), f"{bctx}text must be a string")
        elif block["type"] == "tool_use":
            require_keys(block, ["id", "name", "input"], bctx)
            require(isinstance(block["input"], dict), f"{bctx}input must be an object")
        elif block["type"] == "thinking":
            require_keys(block, ["thinking"], bctx)
        else:
            raise AssertionError(f"{bctx}unexpected content block type {block['type']!r}")

    require(
        body["stop_reason"] in ("end_turn", "max_tokens", "stop_sequence", "tool_use", None),
        f"{ctx}unexpected stop_reason {body['stop_reason']!r}",
    )
    u = body["usage"]
    require_keys(u, ["input_tokens", "output_tokens"], ctx + "usage ")


def validate_anthropic_error(body, status_code):
    ctx = f"[anthropic error, status {status_code}] "
    require_keys(body, ["type", "error"], ctx)
    require(body["type"] == "error", f"{ctx}top-level type must equal 'error'")
    err = body["error"]
    require_keys(err, ["type", "message"], ctx + "error ")


VALID_ANTHROPIC_SSE_EVENTS = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
}


def validate_anthropic_sse_event(event_type, data):
    ctx = f"[anthropic stream event '{event_type}'] "
    require(event_type in VALID_ANTHROPIC_SSE_EVENTS, f"{ctx}not a recognized Anthropic SSE event type")
    if event_type == "message_start":
        require_keys(data, ["message"], ctx)
        require(data["message"]["type"] == "message", f"{ctx}message.type must be 'message'")
    elif event_type == "content_block_start":
        require_keys(data, ["index", "content_block"], ctx)
    elif event_type == "content_block_delta":
        require_keys(data, ["index", "delta"], ctx)
    elif event_type == "message_delta":
        require_keys(data, ["delta"], ctx)
    elif event_type == "error":
        require_keys(data, ["error"], ctx)
