# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Route.LLM is a lightweight Python proxy that translates between the Anthropic Messages API and OpenAI Chat Completions API, allowing tools like Claude Code to route requests to any LLM backend (OpenRouter, Anthropic, OpenAI, local servers) without changing the client. Backend providers can be hot-swapped via a web dashboard without restarting client sessions.

## Commands

```bash
# Run the server (dev mode with auto-reload)
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest test_proxy.py

# Windows quick-start (auto-creates venv, installs deps, starts server)
start_server.bat
```

## Architecture

### Request Flow

1. Client sends request to `/v1/messages` (Anthropic) or `/v1/chat/completions` (OpenAI)
2. `main.py` reads the active provider from SQLite (`database.py`)
3. `translator.py` converts between protocols if the incoming format differs from the backend
4. For streaming: `stream_generator()` in `main.py` performs real-time SSE token-by-token translation between Anthropic and OpenAI streaming event formats, including tool call streaming
5. For non-streaming: complete JSON response translation via `translator.py`
6. All requests/responses logged to SQLite via `database.py`

### Four Translation Scenarios

- Anthropic-in → OpenAI-out (e.g., Claude Code → OpenRouter)
- OpenAI-in → Anthropic-out
- Anthropic-in → Anthropic-out (passthrough)
- OpenAI-in → OpenAI-out (passthrough)

### Key Files

- **`main.py`** — FastAPI app, all route handlers, proxy logic, SSE stream translation (`stream_generator()` ~250 lines handles the complex streaming state machine)
- **`translator.py`** — Protocol translation functions (request and response conversion between Anthropic ↔ OpenAI formats)
- **`database.py`** — SQLite layer for providers, settings, and request logs
- **`test_proxy.py`** — Pytest unit tests for translator and database
- **`static/`** — Single-page dashboard app (providers CRUD, request logs, chat testbed)
- **`proxy.db`** — Runtime SQLite database (gitignored)

### Database Schema (SQLite, `proxy.db`)

- `providers` — id, name, api_type (`"openai"` | `"anthropic"`), endpoint_url, api_key, model_name, is_active
- `settings` — key-value pairs (default: `log_limit` = 50; set to `-1` to disable logging)
- `logs` — request/response audit log with timestamp, provider, method, path, body, status, response

### API Endpoints

**Proxy (client-facing):**
- `POST /v1/messages` — Anthropic Messages API proxy
- `POST /v1/chat/completions` — OpenAI Chat Completions API proxy

**Management (dashboard):**
- `GET/POST /api/providers` — list/create providers
- `PUT /api/providers/{id}` — update provider
- `DELETE /api/providers/{id}` — delete provider
- `POST /api/providers/{id}/active` — set active provider
- `GET/POST /api/settings` — get/set settings
- `GET/DELETE /api/logs` — get/clear request logs
- `POST /api/chat` — test chat from dashboard

## Tech Stack

Python 3, FastAPI, uvicorn, httpx, SQLite (stdlib `sqlite3`), vanilla JS frontend. Only 3 dependencies — the project is intentionally minimal.

## Important Rules
- **API Security**: Always return the `api_key` in the `ProviderResponse` model. Do not exclude it from responses.
