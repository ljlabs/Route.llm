# Integration Tests

This directory contains tests that exercise a running router or coordinate multiple local services. It is intentionally separate from fast, in-process tests under `tests/` and performance tooling under `load_test/`.

## Layout

- `alignment/` — OpenAI Chat Completions and Anthropic Messages protocol-conformance tests against a running router.
- `run_integration.py` — starts a mock backend and proxy for the chat load scenario; its `--mode embedding` checks an already-running proxy and mock.
- `kill_servers.py` — Windows utility to clean up interrupted integration processes on ports 8000, 8081, and 9001.

## Run protocol conformance tests

Start the router with a compatible active provider, then run from the repository root:

```powershell
$env:BASE_URL = 'http://127.0.0.1:8001'
..\.venv\Scripts\python.exe -m pytest integration_tests/alignment -n auto
```

The suite uses `BASE_URL`, `MODEL`, `ANTHROPIC_VERSION`, and `REQUEST_TIMEOUT_SECONDS`. Client API-key headers are accepted but ignored because the local proxy does not authenticate callers. See `alignment/README.md` for markers and endpoint coverage.

## Run the chat integration/load scenario

```powershell
.\.venv\Scripts\python.exe integration_tests/run_integration.py
```

The runner starts `load_test/mock_server.py` and a proxy on port 8000, configures the mock provider, then invokes `load_test/load_test.py`. Do not manually start another mock server on port 9001 for this command.

## Run embedding integration checks

```powershell
.\.venv\Scripts\python.exe integration_tests/run_integration.py --mode embedding
```

Embedding mode expects a proxy on port 8000 and a mock server on port 9001 to already be running.
