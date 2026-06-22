"""
Integration Test Orchestrator

Supports two modes:
  --mode chat      (default) Start mock server + proxy, run chat load test
  --mode embedding           Start mock embedding server + embedding proxy, run embedding integration test
"""

import asyncio
import time
import argparse
import httpx
import os
import sys
import statistics
from typing import List

MOCK_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_server.py")
LOAD_TEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_test.py")

# Ports
MOCK_PORT = 9001
PROXY_PORT = 8000
EMBEDDING_PROXY_PORT = 8081


async def start_process(command: List[str], name: str):
    """Start a process and return its handle."""
    print(f"Starting {name}...")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return proc


async def wait_for_server(url: str, timeout: float = 20.0):
    """Poll until a server responds or timeout."""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Chat load test mode (original behaviour)
# ---------------------------------------------------------------------------

async def run_chat(args):
    processes = []
    try:
        # 1. Start Mock Server (port 9001)
        mock_proc = await start_process(
            [sys.executable, MOCK_SERVER_PATH, "--port", str(MOCK_PORT)], "Mock Server"
        )
        processes.append(mock_proc)

        # 2. Start Proxy Server (port 8000)
        proxy_proc = await start_process(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "127.0.0.1", "--port", str(PROXY_PORT), "--no-reload"],
            "Proxy Server"
        )
        processes.append(proxy_proc)

        # Wait for servers
        print("Waiting for servers...")
        mock_ok = await wait_for_server(f"http://127.0.0.1:{MOCK_PORT}/health")
        proxy_ok = await wait_for_server(f"http://127.0.0.1:{PROXY_PORT}/api/providers")
        if not mock_ok or not proxy_ok:
            print("ERROR: Servers failed to start")
            return
        print("Servers ready.\n")

        # 3. Configure Proxy
        async with httpx.AsyncClient() as client:
            print("Configuring proxy...")

            await client.post(
                f"http://127.0.0.1:{PROXY_PORT}/api/settings",
                json={"rate_limit_tps": args.global_tps}
            )

            provider_data = {
                "name": "MockProvider",
                "api_type": "openai",
                "endpoint_url": f"http://127.0.0.1:{MOCK_PORT}/v1/chat/completions",
                "api_key": "sk-mock",
                "model_name": "mock-model",
                "is_active": 1,
                "rate_limit_tps": args.per_provider_tps
            }
            resp = await client.post(
                f"http://127.0.0.1:{PROXY_PORT}/api/providers",
                json=provider_data
            )
            if resp.status_code >= 400:
                print(f"Error configuring provider: {resp.text}")
                return
            print("Proxy configured.\n")

        # 4. Run Load Test
        print(">>> Starting Load Test <<<\n")
        load_cmd = [
            sys.executable, LOAD_TEST_PATH,
            "--proxy-url", f"http://127.0.0.1:{PROXY_PORT}",
            "--max-tps", str(args.max_tps),
            "--hold-duration", str(args.hold_duration),
            "--ramp-step", str(args.ramp_step),
            "--ramp-interval", str(args.ramp_interval),
            "--concurrency", str(args.concurrency),
        ]

        load_proc = await asyncio.create_subprocess_exec(*load_cmd, stdout=None, stderr=None)
        await load_proc.wait()

    except Exception as e:
        print(f"Integration test error: {e}")
    finally:
        print("\nCleaning up processes...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        print("Done.")


# ---------------------------------------------------------------------------
# Embedding integration test mode
# ---------------------------------------------------------------------------

def _check(name: str, ok: bool, detail: str, passed: list, failures: list):
    if ok:
        print(f"  [PASS] {name}{(' — ' + detail) if detail else ''}")
        passed.append(name)
    else:
        print(f"  [FAIL] {name}: {detail}")
        failures.append(name)


async def run_embedding(args):
    """
    Embedding integration test — runs against already-live servers:
      - Main proxy on PROXY_PORT (8000), which includes /v1/embeddings
      - Mock backend on MOCK_PORT (9001), which serves /v1/embeddings

    Registers an embedding provider on the proxy pointing at the mock, then:
      - Single-query test  : one string, input_type=query semantics
      - Batch ingestion    : list of strings, passage/ingestion semantics
      - Throughput load    : N concurrent single requests
      - Determinism check  : same input → same vector
      - Logging check      : requests appear in /api/logs
      - Mock stats check   : mock tracked the calls
    """
    passed = []
    failures = []
    proxy_base = f"http://127.0.0.1:{PROXY_PORT}"
    mock_embed_url = f"http://127.0.0.1:{MOCK_PORT}/v1/embeddings"
    provider_id = None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:

            # ------------------------------------------------------------------
            # Setup: register embedding provider on the main proxy
            # ------------------------------------------------------------------
            print("Configuring embedding provider on proxy...")
            resp = await client.post(f"{proxy_base}/api/providers", json={
                "name": "MockEmbedding",
                "api_type": "embedding",
                "endpoint_url": mock_embed_url,
                "api_key": "sk-mock",
                "model_name": "mock-embedding-model",
                "is_active": 1,
            })
            if resp.status_code >= 400:
                print(f"ERROR configuring provider: {resp.text}")
                return

            providers = (await client.get(f"{proxy_base}/api/providers")).json()
            provider_id = next(
                (p["id"] for p in reversed(providers) if p["name"] == "MockEmbedding"),
                None
            )
            await client.post(f"{proxy_base}/api/providers/{provider_id}/active")
            print(f"Provider configured (id={provider_id}).\n")

            print(f">>> Running Embedding Integration Tests <<<\n")

            # ------------------------------------------------------------------
            # Test 1: Single query input (input_type=query style)
            # ------------------------------------------------------------------
            resp = await client.post(
                f"{proxy_base}/v1/embeddings",
                json={"model": "mock-embedding-model", "input": "What is the capital of France?"}
            )
            if resp.status_code == 200:
                body = resp.json()
                ok = (
                    body.get("object") == "list"
                    and len(body.get("data", [])) == 1
                    and isinstance(body["data"][0]["embedding"], list)
                    and len(body["data"][0]["embedding"]) == args.embedding_dims
                    and body["data"][0]["index"] == 0
                    and "usage" in body
                )
                _check(
                    "single query input",
                    ok,
                    f"dims={len(body['data'][0]['embedding'])}" if ok else f"body={body}",
                    passed, failures
                )
            else:
                _check("single query input", False,
                       f"HTTP {resp.status_code} — {resp.text[:200]}", passed, failures)

            # ------------------------------------------------------------------
            # Test 2: Batch ingestion input (list of passages)
            # ------------------------------------------------------------------
            passages = [
                "Paris is the capital of France.",
                "The Eiffel Tower was built in 1889.",
                "French cuisine is renowned worldwide.",
                "The Louvre is the world's largest art museum.",
                "France borders Spain, Italy, and Germany.",
            ]
            resp = await client.post(
                f"{proxy_base}/v1/embeddings",
                json={"model": "mock-embedding-model", "input": passages}
            )
            if resp.status_code == 200:
                body = resp.json()
                data = body.get("data", [])
                ok = (
                    body.get("object") == "list"
                    and len(data) == len(passages)
                    and all(len(d["embedding"]) == args.embedding_dims for d in data)
                    and [d["index"] for d in data] == list(range(len(passages)))
                    and "usage" in body
                )
                _check(
                    "batch ingestion input",
                    ok,
                    f"{len(data)} embeddings, dims={len(data[0]['embedding'])}" if ok
                    else f"got {len(data)} items, expected {len(passages)}",
                    passed, failures
                )
            else:
                _check("batch ingestion input", False,
                       f"HTTP {resp.status_code} — {resp.text[:200]}", passed, failures)

            # ------------------------------------------------------------------
            # Test 3: Determinism — same input produces same vector
            # ------------------------------------------------------------------
            r1 = await client.post(f"{proxy_base}/v1/embeddings",
                                   json={"model": "mock-embedding-model", "input": "determinism check"})
            r2 = await client.post(f"{proxy_base}/v1/embeddings",
                                   json={"model": "mock-embedding-model", "input": "determinism check"})
            if r1.status_code == 200 and r2.status_code == 200:
                v1 = r1.json()["data"][0]["embedding"]
                v2 = r2.json()["data"][0]["embedding"]
                _check("deterministic embeddings", v1 == v2,
                       "identical inputs produce identical vectors" if v1 == v2
                       else "vectors differ between identical inputs",
                       passed, failures)
            else:
                _check("deterministic embeddings", False, "requests failed", passed, failures)

            # ------------------------------------------------------------------
            # Test 4: Distinct inputs produce distinct vectors
            # ------------------------------------------------------------------
            ra = await client.post(f"{proxy_base}/v1/embeddings",
                                   json={"model": "mock-embedding-model", "input": "apple"})
            rb = await client.post(f"{proxy_base}/v1/embeddings",
                                   json={"model": "mock-embedding-model", "input": "banana"})
            if ra.status_code == 200 and rb.status_code == 200:
                va = ra.json()["data"][0]["embedding"]
                vb = rb.json()["data"][0]["embedding"]
                _check("distinct inputs produce distinct vectors", va != vb,
                       "vectors differ as expected" if va != vb else "got identical vectors for different inputs",
                       passed, failures)
            else:
                _check("distinct inputs produce distinct vectors", False, "requests failed", passed, failures)

            # ------------------------------------------------------------------
            # Test 5: Throughput — single-query load
            # ------------------------------------------------------------------
            print(f"\n  Running throughput test ({args.num_requests} single-query requests, "
                  f"concurrency={args.concurrency})...")
            latencies = []
            errors = 0
            sem = asyncio.Semaphore(args.concurrency)

            async def _send_query(i: int):
                async with sem:
                    t0 = asyncio.get_event_loop().time()
                    r = await client.post(
                        f"{proxy_base}/v1/embeddings",
                        json={"model": "mock-embedding-model",
                              "input": f"query sentence number {i}"}
                    )
                    return r.status_code, (asyncio.get_event_loop().time() - t0) * 1000

            results = await asyncio.gather(
                *[_send_query(i) for i in range(args.num_requests)],
                return_exceptions=True
            )
            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                else:
                    status, lat = r
                    if status == 200:
                        latencies.append(lat)
                    else:
                        errors += 1

            if latencies:
                p50 = statistics.median(latencies)
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                _check(
                    f"single-query throughput ({args.num_requests} reqs)",
                    errors == 0,
                    f"{len(latencies)} ok, {errors} err | p50={p50:.0f}ms p95={p95:.0f}ms avg={statistics.mean(latencies):.0f}ms",
                    passed, failures
                )
            else:
                _check(f"single-query throughput ({args.num_requests} reqs)", False,
                       f"all {errors} requests failed", passed, failures)

            # ------------------------------------------------------------------
            # Test 6: Throughput — batch ingestion load
            # ------------------------------------------------------------------
            batch_size = 10
            print(f"\n  Running batch ingestion load ({args.num_requests} batches × "
                  f"{batch_size} passages, concurrency={args.concurrency})...")
            batch_latencies = []
            batch_errors = 0

            async def _send_batch(i: int):
                async with sem:
                    t0 = asyncio.get_event_loop().time()
                    batch = [f"passage {i} sentence {j}" for j in range(batch_size)]
                    r = await client.post(
                        f"{proxy_base}/v1/embeddings",
                        json={"model": "mock-embedding-model", "input": batch}
                    )
                    return r.status_code, (asyncio.get_event_loop().time() - t0) * 1000, r

            batch_results = await asyncio.gather(
                *[_send_batch(i) for i in range(args.num_requests)],
                return_exceptions=True
            )
            for r in batch_results:
                if isinstance(r, Exception):
                    batch_errors += 1
                else:
                    status, lat, resp_obj = r
                    if status == 200:
                        body = resp_obj.json()
                        if len(body.get("data", [])) == batch_size:
                            batch_latencies.append(lat)
                        else:
                            batch_errors += 1
                    else:
                        batch_errors += 1

            if batch_latencies:
                p50 = statistics.median(batch_latencies)
                p95 = sorted(batch_latencies)[int(len(batch_latencies) * 0.95)]
                _check(
                    f"batch ingestion throughput ({args.num_requests}×{batch_size})",
                    batch_errors == 0,
                    f"{len(batch_latencies)} ok, {batch_errors} err | p50={p50:.0f}ms p95={p95:.0f}ms avg={statistics.mean(batch_latencies):.0f}ms",
                    passed, failures
                )
            else:
                _check(f"batch ingestion throughput ({args.num_requests}×{batch_size})", False,
                       f"all {batch_errors} batches failed", passed, failures)

            # ------------------------------------------------------------------
            # Test 7: Requests are logged
            # ------------------------------------------------------------------
            logs_resp = await client.get(f"{proxy_base}/api/logs")
            if logs_resp.status_code == 200:
                logs = logs_resp.json()
                embed_logs = [l for l in logs if l.get("request_path") == "/v1/embeddings"]
                _check("requests are logged", len(embed_logs) > 0,
                       f"{len(embed_logs)} log entries found" if embed_logs
                       else "no /v1/embeddings entries in logs",
                       passed, failures)
            else:
                _check("requests are logged", False,
                       f"logs API returned {logs_resp.status_code}", passed, failures)

            # ------------------------------------------------------------------
            # Test 8: Mock stats reflect all embedding calls
            # ------------------------------------------------------------------
            stats_resp = await client.get(f"http://127.0.0.1:{MOCK_PORT}/stats")
            if stats_resp.status_code == 200:
                stats_body = stats_resp.json()
                n = stats_body.get("embedding_requests", 0)
                _check("mock server tracks embedding requests", n > 0,
                       f"{n} embedding requests recorded", passed, failures)
            else:
                _check("mock server tracks embedding requests", False,
                       f"stats returned {stats_resp.status_code}", passed, failures)

        # Summary
        total = len(passed) + len(failures)
        print(f"\n{'='*60}")
        print(f"  EMBEDDING INTEGRATION TEST SUMMARY")
        print(f"  {len(passed)}/{total} passed", end="")
        if failures:
            print(f"  —  {len(failures)} FAILED: {', '.join(failures)}")
        else:
            print("  — ALL PASSED")
        print(f"{'='*60}\n")

    except Exception as e:
        import traceback
        print(f"Embedding integration test error: {e}")
        traceback.print_exc()
    finally:
        # Clean up the provider we registered so we don't pollute the DB
        if provider_id is not None:
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    await c.delete(f"{proxy_base}/api/providers/{provider_id}")
                    print(f"Cleaned up provider id={provider_id}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_integration():
    parser = argparse.ArgumentParser(description="Integration Test Orchestrator")
    parser.add_argument(
        "--mode", choices=["chat", "embedding"], default="chat",
        help="Test mode: 'chat' (load test) or 'embedding' (embedding integration test)"
    )

    # Chat mode flags
    parser.add_argument("--global-tps", type=float, default=0,
                        help="[chat] Global TPS limit to set on proxy")
    parser.add_argument("--per-provider-tps", type=float, default=None,
                        help="[chat] Per-provider TPS limit")
    parser.add_argument("--max-tps", type=float, default=20.0,
                        help="[chat] Max TPS to ramp to")
    parser.add_argument("--hold-duration", type=float, default=30.0,
                        help="[chat] Hold duration in seconds")
    parser.add_argument("--ramp-step", type=float, default=1.0,
                        help="[chat] TPS ramp step")
    parser.add_argument("--ramp-interval", type=float, default=2.0,
                        help="[chat] Ramp interval in seconds")

    # Shared flags
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Max concurrent connections")

    # Embedding mode flags
    parser.add_argument("--num-requests", type=int, default=100,
                        help="[embedding] Number of throughput requests to fire")
    parser.add_argument("--mock-latency-ms", type=int, default=10,
                        help="[embedding] Simulated backend latency in ms (informational only "
                             "when servers are pre-started)")
    parser.add_argument("--embedding-dims", type=int, default=768,
                        help="[embedding] Expected embedding vector dimensions")

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])

    if args.mode == "embedding":
        # Expects proxy (8000) and mock (9001) to already be running
        print(f"Embedding integration test — proxy=127.0.0.1:{PROXY_PORT}, "
              f"mock=127.0.0.1:{MOCK_PORT}\n")
        await run_embedding(args)
    else:
        await run_chat(args)


if __name__ == "__main__":
    asyncio.run(run_integration())
