# Plan: Embedding Model Support for Route.LLM

## Context

The user needs embedding model support (specifically `gemini-embedding-2` via Google AI Studio) alongside the existing LLM router. The embedding flow is fundamentally different from chat — no message history, no Anthropic/OpenAI translation pipeline, no streaming — so it warrants a separate FastAPI app on port 8081 sharing the same database and infrastructure. All existing TPS rate limiting, logging, and metrics infrastructure must be reused.

---

## New Files

### 1. `core/providers/embedding.py` — EmbeddingProvider class
- Extends `BaseProvider` with sensible no-ops for chat-only methods (wrap_request becomes an embedding-specific pass-through, unwrap_response is passthrough, get_stream_translator returns None, requires_translation returns False)
- `wrap_request()`: Takes a dict with `model` and `input` (str or list of str), substitutes `self.model_name`, passes through
- `get_headers()`: Same as OpenAIProvider — `Authorization: Bearer <key>`
- Registered in factory as `api_type = "embedding"`

### 2. `core/embedding/__init__.py` — Empty package init

### 3. `core/embedding/router.py` — EmbeddingRouterService
- Similar structure to `core/router.py` but simplified (no streaming, no Anthropic internal format)
- `route_embedding_request(request_dict)`:
  1. Find active embedding provider (`api_type == "embedding"`) via `ProviderService`
  2. Apply per-provider TPS rate limiting (same `PerProviderRateLimiter`)
  3. `provider.wrap_request()` → POST to `provider.endpoint_url` → log with latency
- Token usage extraction from `usage.prompt_tokens` / `usage.total_tokens` in response
- Logs to `db.add_log()` with `request_path="/v1/embeddings"` — shows up in existing metrics automatically

### 4. `api/embeddings.py` — FastAPI router
- `POST /v1/embeddings` endpoint using `EmbeddingRequest` model
- Delegates to `EmbeddingRouterService`

### 5. `embedding_main.py` — Separate FastAPI app
- Runs on port 8081, shares `database.py` and infrastructure
- Reads `DB_PATH` from env var `EMBEDDING_DB_PATH` (defaults to `proxy.db`) for integration test isolation
- Same startup pattern as `main.py`: init DB, http_client, rate_limiter, per_provider_limiter
- CORS enabled for cross-origin dashboard requests from port 8000
- Serves its own minimal static page at `/` (or the main dashboard can be embedded)

### 6. `start_embedding_server.bat` — Launch script for port 8081

### 7. `tests/test_embedding.py` — Unit + integration tests

---

## Modified Files

### 8. `core/providers/factory.py`
- Import `EmbeddingProvider` and add `"embedding": EmbeddingProvider` to `_PROVIDER_MAP`

### 9. `models/request.py`
- Add `EmbeddingRequest` Pydantic model:
  ```python
  class EmbeddingRequest(BaseModel):
      model: str
      input: Union[str, List[str]]
      encoding_format: Optional[str] = "float"
  ```

### 10. `core/providers/service.py`
- Add `get_active_embedding_provider()` method that filters `db.get_providers()` for `api_type == "embedding"` and `is_active == 1`, then creates via factory

### 11. `static/index.html`
- Add "Embeddings" nav button and tab section (`#tab-embeddings`) with:
  - Model name input
  - Text input area
  - File upload (accepts text, images, audio, video, PDF)
  - "Generate Embeddings" button
  - Response panel (vector dimensions, token usage, raw JSON)
- Add `"embedding"` option to the provider creation modal's API type dropdown

### 12. `static/app.js`
- `sendEmbeddingRequest()` — reads text or file, POSTs to `http://127.0.0.1:8081/v1/embeddings`, renders response
- `renderEmbeddingResponse(data)` — shows vector count, dimensions, token usage, collapsible raw JSON
- `readFileAsText(file)` / `readFileAsBase64(file)` — file reading helpers
- Update `suggestBaseUrl()` to handle `"embedding"` type → `https://generativelanguage.googleapis.com/v1beta/openai/v1/embeddings`

### 13. `static/style.css`
- Styles for embedding tab: file upload zone, response panel, result cards

---

## Integration Test Design (`tests/test_embedding.py`)

```
Module-scoped fixture "embedding_server":
  1. Delete integ_test.db if it exists
  2. Set env var EMBEDDING_DB_PATH → integ_test.db
  3. Launch: python -m uvicorn embedding_main:app --port 8081
  4. Poll /docs until ready (timeout 10s)
  5. yield
  6. Terminate process, delete integ_test.db
```

Unit tests (no server needed):
- `test_embedding_provider_factory()` — ProviderFactory creates EmbeddingProvider from config
- `test_embedding_wrap_request()` — single string input
- `test_embedding_wrap_request_list()` — list input
- `test_embedding_unwrap_response()` — passthrough verification
- `test_embedding_headers()` — correct Authorization header
- `test_embedding_request_model()` — EmbeddingRequest Pydantic validation

Integration tests (server on 8081 with integ_test.db):
- `test_create_embedding_provider()` — POST provider via main server API
- `test_embedding_endpoint_returns_vectors()` — POST /v1/embeddings, verify response structure
- `test_embedding_request_logged()` — verify log entry created with latency_ms > 0
- `test_embedding_rate_limiting()` — verify per-provider TPS is enforced
- `test_embedding_error_handling()` — invalid API key returns proper error

---

## Implementation Order

1. Core: `core/providers/embedding.py` + factory registration + `models/request.py`
2. Router: `core/embedding/__init__.py` + `core/embedding/router.py`
3. API + app: `api/embeddings.py` + `embedding_main.py`
4. Tests: `tests/test_embedding.py` (unit tests first, integration tests with server fixture)
5. UI: index.html, app.js, style.css for embedding tab + provider dropdown
6. Polish: `start_embedding_server.bat`, CLAUDE.md update

## Verification

1. Run `pytest tests/test_embedding.py` — all unit tests pass
2. Run `pytest tests/test_embedding.py -k test_embedding_server` — integration tests launch server, run, clean up
3. Run `pytest` — all existing tests still pass (no regressions)
4. Manual: launch both servers, create embedding provider in dashboard, use Embeddings tab to test
