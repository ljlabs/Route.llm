"""
Mock OpenAI-Compatible Server

A lightweight FastAPI server that mimics an OpenAI chat completions endpoint.
Returns configurable generated text for every request.
"""

import asyncio
import time
import uuid
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Mock OpenAI Server")

# Stats tracking
_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "start_time": None,
}

# Configuration (set via CLI args)
response_latency_ms = 50
response_tokens = 50


def generate_text(num_tokens: int) -> str:
    """Generate lorem-ipsum style text of approximately the requested token count."""
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
        "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
        "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
        "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
        "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
        "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
        "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
        "deserunt", "mollit", "anim", "id", "est", "laborum", "atque", "corporis",
        "suscipit", "laboriosam", "nisi", "aliquid", "commodi", "consequatur",
    ]
    num_words = max(1, num_tokens)
    selected = []
    for i in range(num_words):
        selected.append(words[i % len(words)])
    return " ".join(selected)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _stats["total_requests"] += 1

    try:
        body = await request.json()
    except Exception:
        _stats["total_errors"] += 1
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    model = body.get("model", "mock-model")
    max_tokens = body.get("max_tokens", response_tokens)

    # Simulate LLM latency
    if response_latency_ms > 0:
        await asyncio.sleep(response_latency_ms / 1000.0)

    text = generate_text(max_tokens)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": max_tokens,
            "total_tokens": 10 + max_tokens,
        },
    }

    return JSONResponse(content=response)


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": 0,
                "owned_by": "mock",
            }
        ],
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    elapsed = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    return {
        **_stats,
        "elapsed_seconds": round(elapsed, 1),
        "requests_per_second": round(_stats["total_requests"] / elapsed, 2) if elapsed > 0 else 0,
    }


def main():
    global response_latency_ms, response_tokens, _stats

    parser = argparse.ArgumentParser(description="Mock OpenAI-Compatible Server")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--latency-ms", type=int, default=50, help="Simulated response latency in ms")
    parser.add_argument("--tokens", type=int, default=50, help="Approximate response tokens")
    args = parser.parse_args()

    response_latency_ms = args.latency_ms
    response_tokens = args.tokens
    _stats["start_time"] = time.time()

    print(f"Mock OpenAI Server starting on {args.host}:{args.port}")
    print(f"  Response latency: {response_latency_ms}ms")
    print(f"  Response tokens: {response_tokens}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
