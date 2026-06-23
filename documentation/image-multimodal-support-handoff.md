# Image / Multimodal Content Support — Implementation Handoff

**Date:** 2026-06-23
**Scope:** Add full image content support across all four translation scenarios (Anthropic ↔ OpenAI, streaming and non-streaming)

---

## 1. Executive Summary

**Current state:** Image content is silently dropped by every translation path in the proxy. There are zero code paths that handle `image` blocks (Anthropic format) or `image_url` parts (OpenAI format). The only scenario that works today is passthrough (same-API routing), where the raw stream is forwarded without inspection.

**Goal:** A client sending an image via either the Anthropic Messages API or the OpenAI Chat Completions API should have that image correctly translated and delivered to the backend model, regardless of provider type.

---

## 2. Problem Inventory

### 2.1 `anthropic_to_openai_request()` — Image blocks silently skipped

**Files:**
- `translator.py:92-159`
- `core/providers/translation.py:99-159` (duplicate)

**What happens:** The `for part in content:` loop only matches `text`, `tool_use`, and `tool_result`. An Anthropic image block like:

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBOR..."
  }
}
```

falls through every `elif` branch and is never added to the OpenAI message. No `image_url` part is ever constructed.

**What should happen:** Detect `{"type": "image", "source": {...}}` blocks and convert them to:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,iVBOR..."
  }
}
```

**Important detail:** The OpenAI message `content` field must become a list (not a string) when images are present. Currently, `text_content` is a string and the final message is `{"role": role, "content": text_content}`. When images exist, this must become `{"role": role, "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]}`.

### 2.2 `openai_to_anthropic_request()` — List content stringified, images lost

**Files:**
- `translator.py:218-249`
- `core/providers/translation.py:226-257` (duplicate)

**What happens:** When `content` is a list (OpenAI multimodal format), the code does:

```python
else:
    anth_content = []
    if content:
        anth_content.append({
            "type": "text",
            "text": content   # <-- content is a list here, gets stringified
        })
```

This turns a list like `[{"type": "text", "text": "..."}, {"type": "image_url", ...}]` into a single text block whose `text` field is the stringified list. All image parts are lost.

**What should happen:** When `content` is a list, iterate over its parts:
- `{"type": "text", "text": "..."}` → append `{"type": "text", "text": "..."}` to `anth_content`
- `{"type": "image_url", "image_url": {"url": "image/png;base64,..."}}` → parse the data URI, extract `media_type` and `base64` data, and append `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}}`
- Other part types (e.g., `input_audio`) — pass through or log a warning

### 2.3 `anthropic_to_openai_response()` — Response only handles text and tool_use

**Files:**
- `translator.py:358-402`
- `core/providers/translation.py:349-411` (duplicate)

**What happens:** Only `text` and `tool_use` content blocks are recognized in the response loop. If an Anthropic model ever returns an image block (unlikely today but possible with future models), it would be silently dropped.

**What should happen:** Forward `image` blocks as-is or convert them to the appropriate OpenAI format. Since LLMs rarely return image content in responses, this is lower priority but should be handled for completeness.

### 2.4 `openai_to_anthropic_response()` — Same issue as 2.3

**Files:**
- `translator.py:287-346`
- `core/providers/translation.py:296-355` (duplicate)

**Same pattern:** Only `text` and `tool_use` are handled. Image content in responses would be dropped.

### 2.5 `AnthropicToOpenAIStreamTranslator` — Streaming only forwards text and tool calls

**File:** `core/translation/stream_base.py:80-248`

**What happens:** The translator hardcodes a text `content_block_start` at index 0 (line 130-134) and only processes `delta.content` (text) and `delta.tool_calls`. Any `content_block_start` events for image blocks from the OpenAI upstream would not generate the correct Anthropic SSE events.

**What should happen:** When a non-text `content_block_start` arrives (e.g., for an image), emit the corresponding Anthropic `content_block_start` event with the image block type and content.

### 2.6 `OpenAIToAnthropicStreamTranslator` — Same streaming issue

**File:** `core/translation/stream_base.py:251-336`

**What happens:** Only `content_block_delta` events with text deltas are forwarded. If an Anthropic provider streams an image `content_block_start` or image-related delta, it is completely ignored.

**What should happen:** Forward image content block events through the stream.

### 2.7 `models/response.py` `ContentBlock` — No image fields

**File:** `models/response.py:11-18`

**What happens:**
```python
class ContentBlock(BaseModel):
    type: str = Field(...)
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
```

No `source` field for Anthropic image blocks. Even if translation produced an image block, Pydantic validation would fail or drop the `source` data.

**What should happen:** Add an optional `source` field:
```python
source: Optional[dict] = Field(default=None, description="Image source (Anthropic image blocks)")
```

### 2.8 `models/request.py` `Message` — Accepts any list but downstream fails

**File:** `models/request.py:11-14`

```python
class Message(BaseModel):
    role: str = Field(...)
    content: Union[str, List[Any]] = Field(...)
```

The Pydantic model already accepts `List[Any]` so image parts pass validation. No change needed here, but worth noting.

### 2.9 `sanitize_openai_payload()` — Preserves non-text parts (good)

**Files:**
- `translator.py:7-58`
- `core/providers/translation.py:13-64`

**What happens:** Image parts (`{"type": "image_url", ...}`) are classified as `non_text_parts` and preserved in the list. This is correct behavior — no change needed for this function.

### 2.10 Test coverage — Zero image/multimodal tests

**File:** `tests/test_proxy.py`

No test cases exist for:
- Anthropic image blocks in requests
- OpenAI `image_url` parts in requests
- Image content in responses
- Streaming with image content
- Round-trip translation (Anthropic → OpenAI → Anthropic)

---

## 3. API Format Reference

### Anthropic Image Block (request)

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": "/9j/4AAQSkZJRg..."
  }
}
```

Anthropic also supports URL-based images (less common):
```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "https://example.com/image.jpg"
  }
}
```

### OpenAI Image Part (request)

```json
{
  "type": "image_url",
  "image_url": {
    "url": "image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
}
```

OpenAI also supports URL-based images:
```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.jpg"
  }
}
```

### Data URI Format

`{media_type};base64,{base64_data}`

Parse: split on `;base64,` to get media_type (after ``) and raw base64 data.

---

## 4. Implementation Plan

### Phase 1: Request Translation (non-streaming) — Highest Priority

This is where 95% of real-world usage will hit. Most image sending is in the initial request, not in responses.

#### Change 1: `anthropic_to_openai_request()` in both `translator.py` and `core/providers/translation.py`

**Location:** The `for part in content:` loop (translator.py lines 99-159)

**Current code structure:**
```python
for part in content:
    if not isinstance(part, dict):
        continue
    part_type = part.get("type")
    if part_type == "text":
        text_content += part.get("text", "")
    elif part_type == "tool_use":
        ...
    elif part_type == "tool_result":
        ...
# After loop:
openai_msg = {"role": role, "content": text_content if text_content else None}
```

**Required changes:**

1. Add a new list variable `image_parts = []` alongside `text_content` and `tool_calls`
2. Add a new `elif part_type == "image":` branch that:
   - Extracts `source` from the block
   - If `source.type == "base64"`: constructs `"{media_type};base64,{data}"`
   - If `source.type == "url"`: uses `source.url` directly
   - Appends `{"type": "image_url", "image_url": {"url": data_url}}` to `image_parts`
3. After the loop, when building the final message:
   - If `image_parts` is non-empty, build content as a **list** combining text and image parts
   - If only text (no images), keep existing behavior (string content)
   - If only images (no text), still use list format

**Pseudocode for the message construction after the loop:**
```python
if image_parts:
    content_list = []
    if text_content:
        content_list.append({"type": "text", "text": text_content})
    content_list.extend(image_parts)
    openai_msg = {"role": role, "content": content_list}
else:
    openai_msg = {"role": role, "content": text_content if text_content else None}
```

#### Change 2: `openai_to_anthropic_request()` in both files

**Location:** The `else:` branch for non-system/non-tool messages (translator.py lines 218-249)

**Current code:**
```python
else:
    anth_content = []
    if content:
        anth_content.append({"type": "text", "text": content})
```

**Required changes:**

1. Check if `content` is a list or a string
2. If **string**: keep existing behavior (wrap in text block)
3. If **list**: iterate over parts:
   - `{"type": "text", "text": "..."}` → append `{"type": "text", "text": "..."}` directly
   - `{"type": "image_url", "image_url": {"url": "..."}}` → parse data URI and convert:
     - If URL is a data URI (`...;base64,...`): split to get media_type and base64 data, emit `{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}`
     - If URL is a regular URL: emit `{"type": "image", "source": {"type": "url", "url": "..."}}`
   - Other types: log a warning, skip

**Helper function to add (at top of file):**
```python
def _parse_data_uri(data_url: str) -> tuple[str, str]:
    """Parse a data URI into (media_type, base64_data).
    
    'image/png;base64,iVBOR...' -> ('image/png', 'iVBOR...')
    """
    # Expected format: data:{media_type};base64,{data}
    if not data_url.startswith(""):
        raise ValueError(f"Not a data URI: {data_url[:50]}...")
    
    header, _, data = data_url.partition(";base64,")
    media_type = header.removeprefix("data:")
    return media_type, data
```

### Phase 2: Response Translation (non-streaming)

#### Change 3: `anthropic_to_openai_response()` in both files

**Location:** The `for part in content:` loop (translator.py lines 364-378)

**Required change:** Add an `elif part_type == "image":` branch that preserves image blocks in the response. Since OpenAI responses typically don't include images, this is mainly for passthrough correctness. Convert Anthropic image blocks to OpenAI `image_url` format if the downstream client expects OpenAI format.

#### Change 4: `openai_to_anthropic_response()` in both files

**Location:** The `for part in content:` loop or message processing (translator.py lines 296-311)

**Same approach:** Handle image content blocks if they appear in OpenAI responses. Lower priority since models don't typically return images.

### Phase 3: Streaming Translators

#### Change 5: `AnthropicToOpenAIStreamTranslator` in `core/translation/stream_base.py`

**Location:** Lines 80-248

**Required change:** When receiving OpenAI streaming events that contain image content (e.g., in future multimodal streaming APIs), emit the correct Anthropic `content_block_start` and `content_block_delta` events for image blocks.

For now, the minimum viable change is to not hardcode index 0 as a text block. Track block indices dynamically so image blocks get their own indices.

#### Change 6: `OpenAIToAnthropicStreamTranslator` in `core/translation/stream_base.py`

**Location:** Lines 251-336

**Required change:** When receiving Anthropic streaming events with `content_block_start` for image blocks, forward them as OpenAI-format streaming chunks with image content.

### Phase 4: Models

#### Change 7: `models/response.py` `ContentBlock`

**Add field:**
```python
source: Optional[dict] = Field(default=None, description="Image source block (Anthropic image content)")
```

This ensures Pydantic doesn't reject image content blocks when serializing Anthropic responses.

### Phase 5: Tests

#### Change 8: Test cases in `tests/test_proxy.py`

Add the following test cases:

1. **`test_anthropic_to_openai_request_with_image`** — Send an Anthropic request with a base64 image block, assert the OpenAI request contains the correct `image_url` part with data URI

2. **`test_anthropic_to_openai_request_with_url_image`** — Same but with `source.type: "url"`

3. **`test_anthropic_to_openai_request_mixed_content`** — Message with text + image + tool_use, assert all parts are present in the correct order

4. **`test_openai_to_anthropic_request_with_image_url`** — Send an OpenAI request with `image_url` parts, assert the Anthropic request contains the correct `image` block with base64 source

5. **`test_openai_to_anthropic_request_with_url_image`** — Same but with a regular URL (not data URI)

6. **`test_openai_to_anthropic_request_string_content_unchanged`** — Verify existing string content behavior is preserved (regression test)

7. **`test_image_in_response_translation`** — Anthropic response with image block translates to OpenAI format correctly

8. **`test_round_trip_image_translation`** — Anthropic → OpenAI → Anthropic preserves image data

---

## 5. Edge Cases to Handle

| Edge Case | How to Handle |
|---|---|
| Image block with no `source` | Skip the block, log a warning. Don't crash. |
| `source.type` is neither `base64` nor `url` | Log warning, skip block |
| Malformed data URI (missing `;base64,`) | Try to handle gracefully — if it's just `{type},{data}` without base64 marker, treat data as-is |
| Very large base64 image (>10MB) | Pass through as-is — don't truncate. The provider API will reject if too large |
| Mixed text + images in a single message | Build a list-format content array with text parts and image parts interleaved |
| `tool_result` content containing images | Anthropic allows `tool_result` content to be a list with text and images. The current code only extracts text from `tool_result` content lists (translator.py:133-139). Image data in tool results should also be forwarded. |
| Provider that doesn't support images | The proxy should pass the image through — let the provider API reject it with its own error message. Don't strip images at the proxy level. |

---

## 6. Files to Modify

| File | Changes |
|---|---|
| `translator.py` | Add image handling to `anthropic_to_openai_request()`, `openai_to_anthropic_request()`, both response translators. Add `_parse_data_uri()` helper. |
| `core/providers/translation.py` | Same changes (this is a duplicate — consider consolidating). |
| `core/translation/stream_base.py` | Add image block support to both stream translators. |
| `models/response.py` | Add `source` field to `ContentBlock`. |
| `tests/test_proxy.py` | Add 7-8 new test cases for image content. |

---

## 7. Testing Strategy

### Unit Tests

- Pure translation function tests (no network calls)
- Test both directions: Anthropic → OpenAI and OpenAI → Anthropic
- Test base64 images, URL images, and mixed content
- Test round-trip: Anthropic → OpenAI → Anthropic preserves data
- Regression: existing text-only and tool_use tests still pass

### Integration Tests

- Send an Anthropic request with an image to the proxy, verify the backend receives it in OpenAI format
- Send an OpenAI request with an image to the proxy, verify the backend receives it in Anthropic format
- Use the mock server (`load_test/mock_server.py`) to capture and validate the translated request

### Manual Testing

- Use the dashboard chat testbed to send a message with an image
- Use curl/HTTP client to send both Anthropic and OpenAI format requests with images

---

## 8. Known Risks

1. **Duplicate translation code:** `translator.py` and `core/providers/translation.py` are nearly identical. Changes must be applied to both, or better yet, consolidate them. The version in `core/providers/translation.py` is the one actually imported by provider classes.

2. **Passthrough translator logging:** `PassthroughStreamTranslator` only accumulates text for logs (stream_base.py:62-73). Image data won't appear in logs. This is acceptable — image data in logs would be huge — but worth noting.

3. **No streaming image test infrastructure:** The mock server would need to be updated to generate streaming responses with image content blocks for proper streaming tests.

4. **Provider-specific quirks:** Some OpenAI-compatible providers may not support `image_url` in list content. The proxy should not filter — let the provider reject with its own error.

---

## 9. Acceptance Criteria

- [ ] A client can send an Anthropic request with a base64 image to the proxy, and an OpenAI backend receives the image as an `image_url` data URI
- [ ] A client can send an OpenAI request with an `image_url` to the proxy, and an Anthropic backend receives it as an `image` block with base64 source
- [ ] URL-based images (both formats) are translated correctly
- [ ] Mixed content (text + images + tool_use) is preserved in correct order
- [ ] Existing text-only and tool_use translation tests still pass
- [ ] Streaming with passthrough still works (no regression)
- [ ] At least 7 new unit tests covering image translation paths
