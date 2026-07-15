# Test Suite Layout

- `tests/` contains fast unit, translation, database, and in-process API tests.
- `integration_tests/` contains live router and multi-service integration tests.
- `load_test/` contains only the mock backend, load generator, and generated reports.

## Protocol Conformance

Run the OpenAI and Anthropic alignment suite against an already-running router:

```powershell
$env:BASE_URL = 'http://127.0.0.1:8001'
.\.venv\Scripts\python.exe -m pytest integration_tests/alignment -n auto
```

Useful markers: `openai`, `anthropic`, `streaming`, `tools`, and `vision`. API-key headers are accepted but ignored because the proxy is local-only.

## Local Mock and Load Scenario

```powershell
.\.venv\Scripts\python.exe integration_tests/run_integration.py
```

The runner starts `load_test/mock_server.py` on port 9001, a proxy on port 8000, configures a mock provider, and invokes the load generator. Do not start a duplicate mock server for this command.

## Unit/API Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```
