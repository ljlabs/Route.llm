# Implementation Plan: Stream Translation & Request Handling Fixes

## Overview

This plan addresses verified critical bugs in the model router proxy that cause:
- Silent message/chunk loss during streaming
- Conversation structure corruption (tool results split into separate user messages)
- Tool call data completely dropped in one translation direction
- Error messages injected as model-generated content
- Minor race conditions and naming confusion

All issues have been verified against the source code. Fixes are ordered by impact.

---

## Priority 1: Fix Tool Result Grouping (Biggest Impact)

**File:** `core/providers/translation.py`  
**Function:** `openai_to_anthropic_request()` (lines ~292-317)  
**Problem:** Each OpenAI `tool` role message becomes a separate Anthropic `user` message with one `tool_result`. Anthropic's API expects all tool results from a single assistant turn grouped in ONE `user` message.

**Current behavior:**
```
assistant: [3 tool calls]
user: [tool_result 1]   ← separate message
user: [tool_result 2]   ← separate message  
user: [tool_result 3]   ← separate message
```

**Expected behavior:**
```
assistant: [3 tool calls]
user: [tool_result 1, tool_result 2, tool_result 3]  ← single message
```

**Fix:**
In the `openai_to_anthropic_request` function, accumulate consecutive `tool` role messages into a single `user` message. When iterating through messages:
1. When you encounter a `tool` role message, don't append it immediately
2. Accumulate tool_result blocks in a buffer
3. When you hit a non-tool message (or end of messages), flush the buffer as a single user message containing all accumulated tool_result blocks

**Implementation approach:**
```python
# Replace the current tool handling with:
pending_tool_results = []

for msg in openai_req.get("messages", []):
    role = msg.get("role")
    
    if role == "tool":
        # Accumulate tool results
        tool_result_content = [... build content as before ...]
        pending_tool_results.append({
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id"),
            "content": tool_result_content if tool_result_content else ""
        })
        continue
    
    # Flush pending tool results as a single user message before processing next msg
    if pending_tool_results:
        messages.append({
            "role": "user",
            "content": pending_tool_results
        })
        pending_tool_results = []
    
    # ... handle other roles as before ...

# After loop: flush any remaining tool results
if pending_tool_results:
    messages.append({
        "role": "user",
        "content": pending_tool_results
    })
```

**Note:** The special handling for tool results with images (lines 165-178 in `anthropic_to_openai_request`) may need adjustment too — when there's an image in a tool result, it currently splits into a tool message + user message with the image. This is actually correct for OpenAI→Anthropic direction since OpenAI doesn't support images in tool results. Keep that behavior but ensure the tool_result still gets grouped with other tool_results in the user message.

---

## Priority 2: Replace Silent Exception Swallowing

**File:** `core/translation/stream_base.py`  
**Locations:**
- Line ~99 in `PassthroughStreamTranslator` (inside accumulation logic)
- Line ~228 in `AnthropicToOpenAIStreamTranslator` (main parse loop, catches ALL errors including data loss)
- Line ~302 in `OpenAIToAnthropicStreamTranslator` (main parse loop)

**Problem:** `except Exception: pass` silently drops malformed chunks. The client never knows content was lost. Over long conversations this accumulates causing context drift.

**Fix for each location:**

### PassthroughStreamTranslator (line ~99)
This one is acceptable — it's just for log accumulation, not for data delivery. The raw line is still yielded to the client. **Leave as-is** or add debug logging.

### AnthropicToOpenAIStreamTranslator (line ~228)
This is **critical** — it's the inner try/except that wraps the entire chunk processing logic. If JSON parsing fails or any translation logic throws, the chunk is silently dropped.

**Replace with:**
```python
except json.JSONDecodeError as e:
    logger.warning(f"Malformed SSE chunk (JSON parse error): {e} — raw: {data_content[:200]}")
    # Still yield the raw line as a comment event so the client knows something happened
    continue
except Exception as e:
    logger.error(f"Unexpected error translating stream chunk: {e}", exc_info=True)
    continue
```

### OpenAIToAnthropicStreamTranslator (line ~302)
Same fix:
```python
except json.JSONDecodeError as e:
    logger.warning(f"Malformed SSE chunk (JSON parse error): {e} — raw: {data_content[:200]}")
    continue
except Exception as e:
    logger.error(f"Unexpected error translating stream chunk: {e}", exc_info=True)
    continue
```

---

## Priority 3: Add Tool Call Streaming in Anthropic→OpenAI Direction

**File:** `core/translation/stream_base.py`  
**Class:** `OpenAIToAnthropicStreamTranslator` (lines ~243-311)  
**Problem:** This class translates Anthropic SSE → OpenAI SSE format. It only handles:
- `message_start` → role delta
- `content_block_delta` with `text_delta` type → content delta
- `message_delta` → finish_reason

It completely ignores:
- `content_block_start` with `type: "tool_use"` → should emit tool_calls delta with function name
- `content_block_delta` with `type: "input_json_delta"` → should emit tool_calls delta with arguments
- `content_block_stop` → (used for tracking)

**Fix:** Add handling for tool use blocks. The OpenAI streaming format for tool calls uses:
```json
{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_xxx", "type": "function", "function": {"name": "tool_name", "arguments": ""}}]}}]}
// then argument deltas:
{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"partial"}}]}}]}
```

**Implementation:**
```python
# Add state tracking at the top of translate_stream:
tool_blocks = {}  # Maps Anthropic block index → {id, name, openai_index}
next_tool_index = 0

# Add these event handlers in the main loop:

elif event_type == "content_block_start":
    block = data.get("content_block", {})
    block_index = data.get("index")
    if block.get("type") == "tool_use":
        tool_id = block.get("id", "")
        tool_name = block.get("name", "")
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
    delta = data.get("delta", {})
    delta_type = delta.get("type")
    block_index = data.get("index")
    
    if delta_type == "text_delta":
        # ... existing text handling ...
    elif delta_type == "input_json_delta" and block_index in tool_blocks:
        partial_json = delta.get("partial_json", "")
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
```

Also update the `message_delta` handler to emit `"tool_calls"` as finish_reason when `stop_reason == "tool_use"`.

---

## Priority 4: Fix Error Injection as Model Content

**File:** `core/translation/stream_base.py`  
**Class:** `OpenAIToAnthropicStreamTranslator` (lines ~304-311)

**Problem:** Stream errors are yielded as:
```json
{"choices": [{"delta": {"content": "\n[Stream Error: ...]"}, "finish_reason": "error"}]}
```
The client sees this as model-generated text and includes it in conversation history.

**Fix:** Use a proper OpenAI error format or a clean termination:
```python
except Exception as stream_err:
    logger.error(f"Stream translation error: {stream_err}", exc_info=True)
    # Emit a proper error object, not content
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
```

**Note:** The `AnthropicToOpenAIStreamTranslator` (lines 231-234) already uses a proper error event format (`"type": "error"`) which is correct for Anthropic SSE. No change needed there.

---

## Priority 5: Fix Class Naming Confusion

**File:** `core/translation/stream_base.py`

**Problem:** Class names are backwards:
- `AnthropicToOpenAIStreamTranslator` → actually translates **OpenAI → Anthropic** (takes OpenAI SSE input, produces Anthropic SSE output)
- `OpenAIToAnthropicStreamTranslator` → actually translates **Anthropic → OpenAI** (takes Anthropic SSE input, produces OpenAI SSE output)

**Fix:** Rename both classes:
- `AnthropicToOpenAIStreamTranslator` → `OpenAIToAnthropicStreamTranslator` (or `OpenAIInputToAnthropicOutputStreamTranslator`)
- `OpenAIToAnthropicStreamTranslator` → `AnthropicToOpenAIStreamTranslator` (or `AnthropicInputToOpenAIOutputStreamTranslator`)

**Important:** This requires updating all references. Search for these class names in:
- `core/providers/openai.py` (likely returns one of these from `get_stream_translator`)
- `core/providers/anthropic.py`
- `core/providers/gemini.py`
- `core/providers/mistral.py`
- `core/providers/openrouter.py`
- `core/providers/nvidia_nim.py`
- Any test files

**Approach:** Do a global search for both class names and swap them. Update docstrings to match.

---

## Priority 6: Handle Extended Thinking Blocks

**File:** `core/translation/stream_base.py`  
**Class:** `OpenAIToAnthropicStreamTranslator` (the one that translates Anthropic→OpenAI)

**Problem:** `content_block_delta` events with `type: "thinking_delta"` are silently dropped. Models like Claude with extended thinking enabled return these blocks.

**Fix:** In the `content_block_delta` handler, add:
```python
elif delta_type == "thinking_delta":
    thinking_text = delta.get("thinking", "")
    if thinking_text:
        # Option A: Pass through as content (some OpenAI clients show this)
        # Option B: Emit as a custom field that clients can ignore
        # Recommended: emit as content since OpenAI doesn't have a thinking field
        yield "data: " + json.dumps({
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(__import__('time').time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": thinking_text},
                "finish_reason": None
            }]
        }) + "\n\n"
```

Also in `translation.py` `anthropic_to_openai_response` (non-streaming), add handling for `"thinking"` type content blocks.

---

## Priority 7: Fix Phantom Empty Text Block

**File:** `core/translation/stream_base.py`  
**Class:** `AnthropicToOpenAIStreamTranslator` (the one that translates OpenAI→Anthropic)

**Problem:** A `content_block_start` with empty text is always emitted at index 0, even if the response is purely tool calls.

**Fix:** Defer the text `content_block_start` until actual text content arrives:
```python
# Replace the unconditional content_block_start emission with:
text_block_started = False

# Then when text arrives:
if text and not sent_stop:
    if not text_block_started:
        yield f"event: content_block_start\ndata: " + json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }) + "\n\n"
        text_block_started = True
    # ... yield text delta as before ...
```

Also defer `content_block_stop` for index 0 — only emit it if `text_block_started` is True.

---

## Priority 8: Fix Text Block Collapse in Sanitization

**File:** `core/providers/translation.py`  
**Function:** `sanitize_openai_payload()` (lines ~64-88)

**Problem:** Multiple distinct text blocks in a message's content array are collapsed into a single text block, losing semantic separation.

**Fix:** Only collapse text blocks when sending to strict APIs (Mistral) that require it. For other APIs, preserve the original structure:
```python
# The current behavior IS correct for strict APIs like Mistral
# But the function is applied universally. Add a parameter:
def sanitize_openai_payload(payload: dict, is_gemini: bool = False, collapse_text: bool = True) -> dict:
```

Alternatively, since this function is specifically for "strict APIs", keep the existing behavior but add a comment explaining why. The real fix is to ensure the content ordering is preserved when there ARE non-text parts — currently all text is moved to the front which reorders content. Fix:
```python
elif text_parts or non_text_parts:
    # Preserve original ordering instead of moving all text to front
    new_content = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                new_content.append(part)
            else:
                new_content.append(part)
        elif isinstance(part, str):
            new_content.append({"type": "text", "text": part})
    message["content"] = new_content
```

Wait — that doesn't collapse. The point of this function is to sanitize for strict APIs. The better fix: only collapse when there are ONLY text parts (already done). When mixed, preserve the original order:
```python
elif text_parts or non_text_parts:
    # Keep parts in their original order, just clean them
    new_content = []
    for part in content:
        if isinstance(part, dict):
            part.pop("cache_control", None)
            new_content.append(part)
        elif isinstance(part, str):
            new_content.append({"type": "text", "text": part})
    message["content"] = new_content
```

---

## Priority 9: Provider Cache Race Condition (Low Priority)

**File:** `core/providers/service.py`  
**Method:** `get_active_provider()` (lines ~33-56)

**Problem:** No async locking around cache check/creation. Multiple concurrent requests could create duplicate provider instances.

**Fix:** Add an `asyncio.Lock`:
```python
import asyncio

class ProviderService:
    def __init__(self):
        self._factory = ProviderFactory()
        self._active_provider_cache: Optional[BaseProvider] = None
        self._cache_valid = False
        self._cache_lock = asyncio.Lock()
    
    async def get_active_provider(self) -> Optional[BaseProvider]:
        if self._cache_valid and self._active_provider_cache:
            return self._active_provider_cache
        
        async with self._cache_lock:
            # Double-check after acquiring lock
            if self._cache_valid and self._active_provider_cache:
                return self._active_provider_cache
            # ... rest of method ...
```

**⚠️ WARNING:** This changes the method signature from sync to async. All callers of `get_active_provider()` will need to be updated to `await` it. This is a larger refactor — evaluate if the race condition is actually causing issues in practice before doing this. The current sync approach works fine for single-worker deployments.

**Alternative (simpler):** Leave as sync but document that it's not thread-safe. In practice, Python's GIL + the fact that `db.get_active_provider()` is fast and idempotent means this is unlikely to cause real issues.

---

## Priority 10: Unrecognized Content Blocks Dropping Messages

**File:** `core/providers/translation.py`  
**Function:** `anthropic_to_openai_request()` (lines ~219-238)

**Problem:** If a message's content list contains only unrecognized block types (not text, not image, not tool_use, not tool_result), the condition `text_content or tool_calls or image_blocks` is falsy and the entire message is dropped.

**Fix:** Add a fallback that preserves unrecognized content as text:
```python
# After processing all parts in content list:
if not (text_content or tool_calls or image_blocks):
    # If we processed content blocks but extracted nothing, 
    # serialize the raw content as text to avoid silent drops
    raw_text = json.dumps(content)
    logger.warning(f"Message with unrecognized content blocks preserved as raw text: {raw_text[:200]}")
    messages.append({
        "role": role,
        "content": raw_text
    })
```

---

## Testing Plan

After implementing fixes, verify with:

1. **Unit tests** — `python -m pytest tests/ -v`
2. **Protocol conformance** — `python -m pytest integration_tests/alignment -n auto`
3. **Mock/load orchestration** — `python integration_tests/run_integration.py`
3. **Manual testing scenarios:**
   - Multi-tool-call conversation (verifies fix #1)
   - Long streaming conversation (verifies fix #2)
   - Tool-call-only response with no text (verifies fix #7)
   - Deliberately corrupt a stream chunk to verify logging instead of silent drop (fix #2)
   - Use Anthropic backend with OpenAI client format and trigger tool calls (fix #3)

---

## Files Modified

| File | Changes |
|------|---------|
| `core/providers/translation.py` | Fix tool result grouping, text block ordering, unrecognized block handling |
| `core/translation/stream_base.py` | Replace `except Exception: pass`, add tool call streaming, fix error injection, fix phantom text block, rename classes |
| `core/providers/service.py` | (Optional) Add cache locking |
| `core/providers/*.py` | Update class name references after rename |

---

## Implementation Order

1. **Fix #1** (tool result grouping) — Standalone change in `translation.py`, biggest user-facing impact
2. **Fix #3** (tool call streaming) — Standalone addition in `stream_base.py`
3. **Fix #2** (exception swallowing) — Small targeted change, pair with fix #3
4. **Fix #4** (error injection) — Small change in `stream_base.py`
5. **Fix #5** (class rename) — Do last since it touches many files and is cosmetic
6. **Fixes #6-10** — Lower priority, can be done independently

Each fix should be a separate commit for easy bisection if issues arise.
