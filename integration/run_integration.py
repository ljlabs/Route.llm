"""
Integration Test Orchestrator

Starts the mock server, configures the proxy, and runs the load test.
"""

import asyncio
import time
import argparse
import httpx
import os
import sys
from typing import List

MOCK_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_server.py")
LOAD_TEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_test.py")


async def start_process(command: List[str], name: str):
    """Start a process and return its handle."""
    print(f"Starting {name}...")
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    return proc


async def wait_for_server(url: str, timeout: float = 10.0):
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


async def run_integration():
    parser = argparse.ArgumentParser(description="Integration Test Orchestrator")
    parser.add_argument("--global-tps", type=float, default=0, help="Global TPS limit to set on proxy")
    parser.add_argument("--per-provider-tps", type=float, default=None, help="Per-provider TPS limit")
    parser.add_argument("--max-tps", type=float, default=20.0, help="Max TPS to ramp to")
    parser.add_argument("--hold-duration", type=float, default=30.0, help="Hold duration in seconds")
    parser.add_argument("--ramp-step", type=float, default=1.0, help="TPS ramp step")
    parser.add_argument("--ramp-interval", type=float, default=2.0, help="Ramp interval in seconds")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent connections")

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])

    processes = []
    try:
        # 1. Start Mock Server (port 9001)
        mock_proc = await start_process(
            [sys.executable, MOCK_SERVER_PATH, "--port", "9001"], "Mock Server"
        )
        processes.append(mock_proc)

        # 2. Start Proxy Server (port 8000)
        proxy_proc = await start_process(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--no-reload"],
            "Proxy Server"
        )
        processes.append(proxy_proc)

        # Wait for servers
        print("Waiting for servers...")
        mock_ok = await wait_for_server("http://127.0.0.1:9001/health")
        proxy_ok = await wait_for_server("http://127.0.0.1:8000/api/providers")
        if not mock_ok or not proxy_ok:
            print("ERROR: Servers failed to start")
            return
        print("Servers ready.\n")

        # 3. Configure Proxy
        async with httpx.AsyncClient() as client:
            print("Configuring proxy...")

            await client.post("http://127.0.0.1:8000/api/settings", json={"rate_limit_tps": args.global_tps})

            provider_data = {
                "name": "MockProvider",
                "api_type": "openai",
                "endpoint_url": "http://127.0.0.1:9001/v1/chat/completions",
                "api_key": "sk-mock",
                "model_name": "mock-model",
                "is_active": 1,
                "rate_limit_tps": args.per_provider_tps
            }
            resp = await client.post("http://127.0.0.1:8000/api/providers", json=provider_data)
            if resp.status_code >= 400:
                print(f"Error configuring provider: {resp.text}")
                return
            print("Proxy configured.\n")

        # 4. Run Load Test
        print(">>> Starting Load Test <<<\n")
        load_cmd = [
            sys.executable, LOAD_TEST_PATH,
            "--proxy-url", "http://127.0.0.1:8000",
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
            except:
                pass
        print("Done.")


if __name__ == "__main__":
    asyncio.run(run_integration())
