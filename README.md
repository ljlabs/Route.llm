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

#### How It Works Under the Hood:
1. **Claude Code** dispatches Anthropic protocol `/v1/messages` requests to `http://localhost:8000/v1/messages`.
2. **Route.LLM** intercepts the request and verifies the active provider format.
3. If the active provider is an OpenAI-compatible server (e.g. OpenRouter):
   - It converts the Anthropic tool schemas, messages, and parameters into OpenAI specifications.
   - It fires the request to your configured `endpoint_url`.
   - It intercepts the incoming stream delta tokens, wraps them into Anthropic-formatted stream events, and yields them to Claude Code.

---

## Unit Testing

Run one-off unit tests covering request translations, tool conversions, and SQLite database retention limits using `pytest`:

```bash
# Activate env and run pytest
.venv/Scripts/pytest test_proxy.py
```

---

## Technology Stack

- **Backend**: FastAPI (Python), HTTPX (Async HTTP Client), SQLite (Log database).
- **Frontend**: Single-Page App utilizing HTML5, HSL-based vanilla CSS variables, and modern Javascript.
- **Design System**: Premium dark-mode layout with responsive glassmorphism blur and visibility states.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
