![image](https://raw.githubusercontent.com/ljlabs/Route.llm/refs/heads/main/static/images/banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)

**Routte.LLM** is a lightweight, local LLM proxy and routing engine. It acts as an adaptable translation bridge between the **Anthropic Messages** and **OpenAI Chat Completions** protocols, allowing you to route any AI agent (like **Claude Code**) to your choice of backend models—including OpenRouter, Anthropic, OpenAI, or local servers.

It also allows you to hot swap backend providers without restarting your claude code sessions.
---

## Key Features

- ⚡ **Seamless Stream Translation**: Stateful conversion of SSE (Server-Sent Events) streaming tokens between Anthropic and OpenAI protocols.
- ⚙️ **Robust Tool & Function Calling Support**: Fully translates tool schemas, tool calls, and tool execution results back-and-forth, enabling agentic assistants to execute multi-step routines.
- 🎛 **Dynamic Provider Control**: Register, update, and switch the active backend endpoint on the fly.
- ▤ **Real-Time Request Inspector**: Inspect JSON request bodies, response values, and detailed connection or API error tracebacks inside the dashboard.
- ⏱ **Log Retention Controls**: Define SQLite log rotation boundaries (e.g., limit to the last N logs, or disable completely) to maintain disk space.
- 🧪 **One-Click Local Runner**: Bundled with a simple Windows runner script (`start_server.bat`) that auto-boots virtual environments and handles dependencies.

---

## Quick Start

### 1. Launch the Server

#### On Windows (Automatic)
Simply double-click **`start_server.bat`**. This script will automatically:
1. Create a Python virtual environment (`.venv`) if it doesn't exist.
2. Install all necessary dependencies from `requirements.txt`.
3. Launch the FastAPI Uvicorn server in a persistent console window.

#### On macOS / Linux / Windows (Manual)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the proxy server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open the Dashboard: 🔗 **[http://localhost:8000/](http://localhost:8000/)**

---

## Client Configurations

To route your favorite developer tools and coding agents through **Route.LLM**, simply override their default base URL.

### Pointing Claude Code
Run Claude Code by setting the `ANTHROPIC_BASE_URL` environment variable pointing to your local proxy server:

```powershell
# PowerShell
$env:ANTHROPIC_BASE_URL="http://localhost:8000"
claude

# macOS/Linux Bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
claude
```

## Architecture

Route.LLM is a FastAPI gateway with one public compatibility surface and pluggable upstream providers:

```text
OpenAI or Anthropic client
          |
          v
api/ proxy routes -> core/router.py -> provider service/factory -> configured upstream
          |                    |                 |
          |                    |                 +-- OpenAI, Anthropic, Gemini, OpenRouter, NVIDIA, embeddings
          |                    +-- request/response + SSE translation, rate limits, lifecycle logging
          v
static/ dashboard <----------- database.py (SQLite providers, mappings, settings, request events)
```

- `api/` exposes the OpenAI Chat Completions, Anthropic Messages, models, provider, settings, logs, metrics, and embeddings routes.
- `core/` owns model routing, provider adapters, request/response translation, streaming SSE translation, and rate limiting.
- `models/` validates public request payloads; `infrastructure/` configures shared HTTP clients.
- `database.py` persists provider configuration, model mappings, settings, request logs, and lifecycle events.
- `static/` is the local dashboard; `load_test/mock_server.py` is a deterministic backend for local validation.

For example, Claude Code sends `/v1/messages` to the proxy. The router selects a mapped or active provider, translates the payload when formats differ, calls the upstream endpoint, then translates the response and SSE events back to Anthropic format.

See [the public API compatibility contract](documentation/API_COMPATIBILITY.md) for the supported OpenAI and Anthropic surface, provider-dependent capabilities, and intentional non-goals.

---

## Testing

The repository separates fast checks, live protocol checks, and performance tooling:

| Area | Purpose | Command from repository root |
| --- | --- | --- |
| `tests/` | Unit, translation, database, and in-process API tests | `.venv\\Scripts\\python.exe -m pytest tests -v` |
| `integration_tests/alignment/` | Live OpenAI and Anthropic protocol-conformance tests | `.venv\\Scripts\\python.exe -m pytest integration_tests/alignment -n auto` |
| `integration_tests/` | Multi-service orchestration and Windows cleanup utility | `.venv\\Scripts\\python.exe integration_tests/run_integration.py` |
| `load_test/` | Mock backend, load generator, and ignored HTML reports | `.venv\\Scripts\\python.exe load_test/load_test.py --help` |

The alignment suite targets an existing router. Set `BASE_URL` (for example, `http://127.0.0.1:8001`) before running it. The integration runner starts its own mock backend on port 9001 and proxy on port 8000; do not start a duplicate mock for that command.

See `integration_tests/README.md` and `load_test/README.md` for suite-specific setup and options.

---

## Technology Stack

- **Backend**: FastAPI (Python), HTTPX (Async HTTP Client), SQLite (Log database).
- **Frontend**: Single-Page App utilizing HTML5, HSL-based vanilla CSS variables, and modern Javascript.
- **Design System**: Premium dark-mode layout with responsive glassmorphism blur and visibility states.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
