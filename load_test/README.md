# Load Testing

Test the proxy's rate limiter end-to-end with a fake LLM backend.

## Quick Start

### Simple mode (fire N requests)

```bash
# Configure proxy first (or use run_integration.py)
python integration/load_test.py --proxy-url http://127.0.0.1:8000 \
  --requests 20 --concurrency 10 --tps 5
```

### Ramp mode (find the limit)

```bash
python integration/run_integration.py --max-tps 10 --hold-duration 30
```

Reports are saved to `integration/output/loadtest_*.html`.

## Manual Setup

```bash
# Terminal 1: Mock LLM (port 9001)
python integration/mock_server.py --port 9001

# Terminal 2: Proxy (port 8000)
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 3: Load test
python integration/load_test.py --proxy-url http://127.0.0.1:8000 --requests 20 --concurrency 10
```

## How It Works

**Simple mode** (`--requests N`): Fires N concurrent requests, reports actual TPS vs target, generates HTML report.

**Ramp mode** (no `--requests`): TPS increases incrementally until it matches the proxy's rate limit, holds steady state, then generates HTML report.

## Options

### `load_test.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--proxy-url` | http://127.0.0.1:8000 | Proxy server URL |
| `--requests` | 0 | Total requests to send (simple mode) |
| `--tps` | 0 | Target TPS limit to verify (simple mode) |
| `--concurrency` | 10 | Max concurrent connections |
| `--model` | None | Model name to send in requests |
| `--setup-provider` | None | Add provider: `NAME ENDPOINT_URL RATE_TPS MODEL` (repeatable) |
| `--setup-global-tps` | 0 | Global TPS limit when using --setup-provider |
| `--setup-mapping` | None | Map `MODEL_ID PROVIDER_NAME` (repeatable) |
| `--ramp-step` | 1.0 | TPS increase per ramp interval (ramp mode) |
| `--ramp-interval` | 2.0 | Seconds between ramp steps (ramp mode) |
| `--hold-duration` | 60 | Seconds to hold steady state (ramp mode) |
| `--max-tps` | 100 | Maximum TPS target (ramp mode) |

### `run_integration.py` (orchestrator)

| Flag | Default | Description |
|------|---------|-------------|
| `--global-tps` | 0 | Global rate limit on the proxy (0 = unlimited) |
| `--per-provider-tps` | None | Per-provider rate limit override |
| `--max-tps` | 20 | Maximum TPS to ramp to |
| `--hold-duration` | 30 | Seconds to hold steady state |
| `--ramp-step` | 1.0 | TPS ramp step |
| `--ramp-interval` | 2.0 | Ramp interval in seconds |
| `--concurrency` | 50 | Max concurrent connections |

### `mock_server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 9001 | Server port |
| `--host` | 127.0.0.1 | Bind address |
| `--latency-ms` | 50 | Simulated response latency |
| `--tokens` | 50 | Approximate response tokens |

## Examples

```bash
# Simple: 20 requests, 10 concurrent, verify 5 TPS
python integration/load_test.py --requests 20 --concurrency 10 --tps 5

# Model-specific: send requests with a specific model name
python integration/load_test.py --requests 20 --concurrency 10 --tps 2 --model my-model

# Model routing: set up providers with per-provider rate limits
python integration/load_test.py --requests 20 --concurrency 10 --tps 1 --model model-a \
  --setup-provider DefaultProvider http://127.0.0.1:9001/v1/chat/completions 0 mock-model \
  --setup-provider ModelAProvider http://127.0.0.1:9001/v1/chat/completions 1 mock-model \
  --setup-mapping model-a ModelAProvider \
  --setup-global-tps 2

# Ramp mode: find the actual limit
python integration/load_test.py --max-tps 20 --hold-duration 30

# Full integration: configure proxy + run ramp test
python integration/run_integration.py --global-tps 5 --max-tps 10 --hold-duration 30
```

## HTML Report

Each run generates `integration/output/loadtest_YYYYMMDD_HHMMSS.html` with:

- **Rate limiter verdict** (ACTIVE / NOT ENFORCED)
- **Summary stats** (requests, TPS, latency percentiles)
- **5 charts**: TPS over time, latency over time (avg+p95), latency scatter, histogram, status codes

Open in any browser — no server needed.
