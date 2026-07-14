# Model Router Conformance Suite

A `pytest` + `requests` suite that hits your router's HTTP endpoints and
checks the response shapes against the **OpenAI Chat Completions spec**
and the **Anthropic Messages API spec**. It's meant to catch the ways a
"mostly compatible" server breaks real SDKs: missing fields, wrong
`finish_reason`/`stop_reason` values, malformed streaming events, wrong
error envelope, wrong status codes, etc.

## Setup

```bash
pip install -r requirements.txt
```

Configure via environment variables (both default to `http://localhost:8001`):

```bash
export OPENAI_BASE_URL=http://localhost:8001
export OPENAI_API_KEY=your-test-key
export OPENAI_MODEL=gpt-4o-mini          # a model id your router accepts on the OpenAI-shaped route

export ANTHROPIC_BASE_URL=http://localhost:8001
export ANTHROPIC_API_KEY=your-test-key
export ANTHROPIC_MODEL=claude-sonnet-4-6 # a model id your router accepts on the Anthropic-shaped route
export ANTHROPIC_VERSION=2023-06-01
```

## Run

```bash
pytest                          # everything
pytest -m openai                # only OpenAI-spec tests
pytest -m anthropic              # only Anthropic-spec tests
pytest -m "openai and streaming" # only OpenAI streaming tests
pytest -m tools                  # tool/function calling, both specs
pytest -m errors                 # error format + auth + malformed input
pytest -m vision                 # image/multimodal input
pytest -v test_openai_chat_completions.py::test_max_tokens_is_respected_via_finish_reason
```

Tests make **real calls** against your router (no mocking) — that's the
point, since you want to know your server actually behaves correctly
end-to-end, including whatever's behind it. Expect this to cost real
tokens/time if your router forwards to a live model.

## What's covered

### OpenAI spec (`/v1/chat/completions`, `/v1/models`)
- `test_openai_chat_completions.py` — required/optional fields, `n`,
  `max_tokens`→`finish_reason=length`, `response_format: json_object`,
  content-as-array, multi-turn, system message, 400s for missing
  `model`/`messages`, empty `messages`, invalid `role`, unknown model.
- `test_openai_streaming.py` — SSE content-type, chunk schema, role
  delta on first chunk, `[DONE]` sentinel, `finish_reason` on the final
  chunk, `stream_options.include_usage`.
- `test_openai_tools.py` — tool call triggering + shape, forced
  `tool_choice`, `tool_choice: "none"`, full round-trip with a `tool`
  role message, streamed tool-call argument deltas assembling into
  valid JSON.
- `test_openai_models_and_vision.py` — `GET /v1/models`,
  `GET /v1/models/{id}` (found + 404), image content parts (URL and
  base64 data URI).

### Anthropic spec (`/v1/messages`)
- `test_anthropic_messages.py` — required `max_tokens`, top-level
  `system` (string and block-array form), multi-turn, content as block
  array, `max_tokens`→`stop_reason=max_tokens`, `stop_sequences`, 400s
  for missing `model`/`max_tokens`/`messages`, conversation not
  starting with `user`, `system` role rejected inside `messages[]`,
  401 on missing `x-api-key`.
- `test_anthropic_streaming.py` — event ordering
  (`message_start` → ... → `message_stop`), `text_delta` reassembly,
  `message_delta` carrying `stop_reason` + cumulative `usage`.
- `test_anthropic_tools.py` — `tool_use` triggering + shape, forced
  `tool_choice`, `tool_choice: {"type": "none"}`, tool_result round
  trip, streamed `input_json_delta` assembling into valid JSON.
- `test_anthropic_vision_and_misc.py` — base64 and URL image blocks,
  `/v1/messages/count_tokens` (skipped if not implemented — it's
  optional).

### Cross-cutting (`test_cross_cutting_errors.py`)
Malformed JSON body, wrong/missing API key, wrong HTTP verb, unknown
route, unknown extra fields not causing a 500 — for both specs.

## Validators

`validators.py` has hand-written structural checks (not full JSON
Schema) so a failing assertion tells you exactly which field is
missing or the wrong type/value, e.g.:

```
AssertionError: [openai chat.completion] choices[0] finish_reason unexpected 'incomplete'
```

`sse.py` has minimal parsers for both streaming formats (OpenAI's
`data: {...}` + `[DONE]`, Anthropic's `event:`/`data:` pairs).

## Extending this

This covers the high-value 80% of both specs, not literally every
field combination (e.g. it doesn't sweep every `logprobs`/
`top_logprobs` combination, batch APIs, or every beta header). Natural
places to add more:

- `logprobs` / `top_logprobs` on OpenAI completions
- Anthropic **extended thinking** (`thinking: {type: "enabled", ...}`)
  and the resulting `thinking` content block
- Anthropic prompt caching (`cache_control` blocks) if your router
  passes those through
- Batch/async endpoints if your router exposes them
- Rate-limit (`429`) behavior and `Retry-After` header conformance
- Concurrency/load tests (this suite is correctness-focused, not load)

## A note on what "conformance" means here

Some 400-vs-different-400-message details are judgment calls where
real providers themselves aren't perfectly consistent. Where the spec
is genuinely ambiguous (e.g. exact wording of error messages), these
tests check the **envelope shape** (`error.type`, `error.message`
present) rather than exact string matching, so you're not chasing
provider quirks that don't actually break clients.
