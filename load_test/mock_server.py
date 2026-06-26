"""
Mock OpenAI-Compatible Server

A lightweight FastAPI server that mimics an OpenAI chat completions endpoint,
an OpenAI-compatible embeddings endpoint, and an Anthropic messages endpoint.
Returns configurable generated text / embedding vectors for every request.
"""

import asyncio
import time
import uuid
import argparse
import math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Mock Server")

# Stats tracking
_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "embedding_requests": 0,
    "total_image_requests": 0,
    "start_time": None,
}

# Configuration (set via CLI args)
response_latency_ms = 50
response_tokens = 50
embedding_dims = 768


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


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image content blocks."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image", "image_url"):
                        return True
    return False


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

    # Track image requests
    messages = body.get("messages", [])
    if _has_image_content(messages):
        _stats["total_image_requests"] += 1

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


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic-compatible messages endpoint (for testing Anthropic passthrough)."""
    _stats["total_requests"] += 1

    try:
        body = await request.json()
    except Exception:
        _stats["total_errors"] += 1
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    model = body.get("model", "mock-model")
    max_tokens = body.get("max_tokens", response_tokens)

    # Track image requests
    messages = body.get("messages", [])
    if _has_image_content(messages):
        _stats["total_image_requests"] += 1

    # Simulate LLM latency
    if response_latency_ms > 0:
        await asyncio.sleep(response_latency_ms / 1000.0)

    text = generate_text(max_tokens)
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"

    response = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": max_tokens,
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


def generate_embedding(text: str, dims: int) -> list:
    """Generate a deterministic unit-normalised mock embedding vector."""
    # Use a simple hash-based seed so identical inputs get identical vectors
    seed = hash(text) & 0xFFFFFFFF
    vector = []
    for i in range(dims):
        # Pseudo-random via linear congruential step
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        vector.append((seed / 0xFFFFFFFF) * 2 - 1)
    # L2-normalise
    magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / magnitude, 8) for v in vector]


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    _stats["total_requests"] += 1
    _stats["embedding_requests"] += 1

    try:
        body = await request.json()
    except Exception:
        _stats["total_errors"] += 1
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    raw_input = body.get("input", "")
    model = body.get("model", "mock-embedding-model")

    # Simulate latency
    if response_latency_ms > 0:
        await asyncio.sleep(response_latency_ms / 1000.0)

    # Normalise input to a list of strings
    if isinstance(raw_input, str):
        inputs = [raw_input]
    else:
        inputs = list(raw_input)

    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": generate_embedding(text, embedding_dims),
        }
        for i, text in enumerate(inputs)
    ]

    prompt_tokens = sum(max(1, len(t.split())) for t in inputs)

    return JSONResponse(content={
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
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
    global response_latency_ms, response_tokens, embedding_dims, _stats

    parser = argparse.ArgumentParser(description="Mock OpenAI-Compatible Server")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--latency-ms", type=int, default=50, help="Simulated response latency in ms")
    parser.add_argument("--tokens", type=int, default=50, help="Approximate response tokens")
    parser.add_argument("--embedding-dims", type=int, default=768, help="Embedding vector dimensions")
    args = parser.parse_args()

    response_latency_ms = args.latency_ms
    response_tokens = args.tokens
    embedding_dims = args.embedding_dims
    _stats["start_time"] = time.time()

    print(f"Mock Server starting on {args.host}:{args.port}")
    print(f"  Response latency: {response_latency_ms}ms")
    print(f"  Response tokens: {response_tokens}")
    print(f"  Embedding dims: {embedding_dims}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
