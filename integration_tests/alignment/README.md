# OpenAI and Anthropic Protocol-Conformance Tests

This live HTTP suite verifies that a running router conforms to the OpenAI Chat Completions and Anthropic Messages APIs. It catches missing fields, invalid error envelopes, malformed SSE, streaming termination errors, tool-call incompatibilities, and multimodal request regressions.

## Setup

From the repository root, install the suite dependencies if they are not already available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r integration_tests/alignment/requirements.txt
```

Configure the target router with environment variables:

```powershell
$env:BASE_URL = 'http://127.0.0.1:8001'
# Optional: overrides both protocol defaults.
Remove-Item Env:MODEL -ErrorAction SilentlyContinue
# API-key headers are optional and ignored by the local proxy.
```

Without `MODEL`, OpenAI tests use `gpt-4o-mini` and Anthropic tests use `claude-sonnet-4-6`. The router must map or fall back from those aliases to an active chat provider.

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -n auto
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -m openai -n auto
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -m anthropic -n auto
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -m streaming -n auto
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -m vision -n auto
```

`REQUEST_TIMEOUT_SECONDS` defaults to 30. The suite does not start or configure the router; use `integration_tests/run_integration.py` for the separate local mock/load scenario.
