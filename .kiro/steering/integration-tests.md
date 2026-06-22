# Integration Tests

## Running Integration Tests

The integration test orchestrator is at `load_test/run_integration.py`. It handles everything — starts the mock server, starts the proxy server, configures them, and runs the load test.

```bash
python load_test/run_integration.py
```

Optional flags:
- `--max-tps` (default 20.0) — peak TPS to ramp to
- `--hold-duration` (default 30.0) — seconds to hold at peak
- `--ramp-step` (default 1.0) — TPS increment per step
- `--ramp-interval` (default 2.0) — seconds between ramp steps
- `--concurrency` (default 50) — max concurrent connections
- `--global-tps` (default 0) — global TPS limit on proxy
- `--per-provider-tps` — per-provider TPS limit

## Mock Server

The mock server (`load_test/mock_server.py`) is started automatically by `run_integration.py` on port 9001. Do not start it manually when running integration tests.

## Unit Tests

```bash
python -m pytest tests/ -v
```

The test suite has one known flaky test (`test_embedding_route_with_provider`) that passes in isolation but can timeout when run alongside the full suite due to server startup timing. This is pre-existing and unrelated to code changes.
