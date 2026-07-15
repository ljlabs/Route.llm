# Load Testing

`load_test/` contains performance tooling only: a deterministic mock backend, the chat load generator, and generated reports. Functional and protocol-conformance tests live in `integration_tests/`.

## Contents

- `mock_server.py` — OpenAI-, Anthropic-, and Responses-compatible local backend on port 9001.
- `load_test.py` — concurrent request generator with simple and ramp modes.
- `output/` — generated HTML reports; ignored by Git.

## Manual load test

Start the mock and proxy in separate terminals, then run the generator from the repository root:

```powershell
.\.venv\Scripts\python.exe load_test/mock_server.py --port 9001
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe load_test/load_test.py --proxy-url http://127.0.0.1:8000 --requests 20 --concurrency 10 --tps 5
```

## Automated local scenario

The orchestration script lives with integration tests because it configures multiple services before invoking this load generator:

```powershell
.\.venv\Scripts\python.exe integration_tests/run_integration.py --max-tps 10 --hold-duration 30
```

It starts the mock and proxy, configures a mock provider, runs a ramp test, and cleans up its processes. Do not manually start port 9001 for this command.

## Load generator options

`load_test.py` supports `--proxy-url`, `--requests`, `--tps`, `--concurrency`, `--model`, `--ramp-step`, `--ramp-interval`, `--hold-duration`, and `--max-tps`. Run `python load_test/load_test.py --help` for the complete list.

For protocol compatibility tests, use `integration_tests/alignment/`; for in-process unit and API tests, use `tests/`.
