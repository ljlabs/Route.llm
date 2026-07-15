# Test Suite Layout

- `tests/` contains fast unit, translation, database, and in-process API tests.
- `integration_tests/` contains tests that exercise a running router or coordinate multiple local services.
- `load_test/` contains only performance tooling, the local mock backend, and generated load reports.

## Protocol-Conformance Integration Tests

The OpenAI and Anthropic alignment suite is at `integration_tests/alignment/`. It targets an already-running router; it does not start or configure one.

From the repository root:

```powershell
$env:BASE_URL = 'http://127.0.0.1:8001'
..\.venv\Scripts\python.exe -m pytest integration_tests/alignment -n auto
```

Useful markers: `-m openai`, `-m anthropic`, `-m streaming`, `-m tools`, and `-m vision`. `MODEL` overrides both protocol defaults. Without it, the suite uses `gpt-4o-mini` for OpenAI tests and `claude-sonnet-4-6` for Anthropic tests. API-key headers are accepted but ignored because this is a local-only proxy.

## Local Mock and Load Scenario

`integration_tests/run_integration.py` starts `load_test/mock_server.py`, starts a proxy on port 8000, configures the mock provider, and runs `load_test/load_test.py`.

```powershell
.\.venv\Scripts\python.exe integration_tests/run_integration.py
```

Do not start another mock server on port 9001 for this command. The runner's `--mode embedding` checks already-running services rather than starting them.

## Unit/API Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

The test suite has one known flaky test (`test_embedding_route_with_provider`) that passes in isolation but can time out in the full suite because of server startup timing. This is pre-existing and unrelated to code changes.
