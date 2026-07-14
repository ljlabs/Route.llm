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
    num_words = max(1, min(num_tokens, 20))
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


def _has_pdf_content(messages: list) -> bool:
    """Check if any message contains PDF content blocks."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "document" and part.get("source", {}).get("media_type") == "application/pdf":
                        return True
                    if part.get("type") == "file" and part.get("mime_type") == "application/pdf":
                        return True
        elif isinstance(content, str) and "application/pdf" in content:
            return True
    return False


def _get_recipe_response() -> dict:
    """Return the recipe PDF parsing response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "The user wants me to parse a recipe from an image/PDF and return it as a JSON object following a specific schema.\n\n**1. Identify the Title and Description:**\n- Title: \"Crispy Buffalo Chicken Tacos and Chips\"\n- Description: \"with Cheddar and Honey Mustard Baby Gem Salad\"\n\n**2. Identify Servings:**\n- The ingredients table has columns for 2P, 3P, and 4P. Usually, the base recipe is for 2 people unless specified otherwise. I'll use 2 as the default servings.\n\n**3. Extract Ingredients (for 2P):**\n- Potatoes: 450g\n- Chicken Breast: 2 (unit: pieces/breasts)\n- Panko Breadcrumbs: 50g\n- Central American Style Spice Mix: 1 sachet\n- Mature Cheddar Cheese: 40g\n- Baby Gem Lettuce: 1 (unit: head/piece)\n- Cider Vinegar: 15ml\n- Sriracha: 30g\n- Honey: 15g\n- Creme Fraiche: 75g\n- Honey and Mustard Dressing: 30g\n- Plain Taco Tortilla: 6 (unit: pieces)\n- Egg: 1 (unit: piece)\n- Salt for the Breadcrumbs: 1/4 tsp (0.25)\n- Water for the Sauce: 100ml\n- Mayonnaise: 2 tbsp\n\n**4. Extract Instructions:**\n- Step 1: \"Preheat your oven to 220°C/200°C fan/gas mark 7. Chop the potatoes lengthways into 1cm slices, then chop into 1cm wide chips (no need to peel). Pop the chips onto a large baking tray. Drizzle with oil, season with salt and pepper, then toss to coat. Spread out in a single layer. When the oven is hot, bake on the top shelf until golden, 25-30 mins. Turn halfway through.\"\n- Step 2: \"Meanwhile, sandwich each chicken breast between two pieces of baking paper or cling film. Pop onto a board, then give it a bash with the bottom of a saucepan until it's 1-2 cm thick. Season with salt and pepper. Crack the egg into a medium bowl and whisk. In another medium bowl, combine the panko breadcrumbs, Central American style spice mix and the salt. Dip the chicken into the egg and then the breadcrumbs, ensuring it's completely coated. Transfer to a clean plate.\"\n- Step 3: \"Pop a large frying pan on high heat and add enough oil to coat the bottom of the pan. Once hot, carefully lay the chicken into the pan, and fry until golden brown, 2-3 mins on each side. Adjust the heat if necessary. Transfer the chicken to a large baking tray, then bake on the middle shelf until cooked through, 8-10 mins.\"\n- Step 4: \"While the chicken bakes, grate the Cheddar cheese. Trim the baby gem, halve lengthways, then thinly slice. Carefully discard the hot oil from the chicken pan, then wipe out and return to medium-high heat. Pour in the cider vinegar and water for the sauce. Bring to the boil, then simmer until reduced by half, 2-3 mins. Stir in the sriracha and honey until thickened, 1-2 mins.\"\n- Step 5: \"Once your buffalo sauce has thickened, stir through the creme fraiche until piping hot, 1-2 mins. Season with salt and pepper. Add a splash of water if it's a little too thick, then remove from the heat. When the chips have 5 mins left, in a medium bowl, combine the baby gem with the honey and mustard dressing. Season with salt and pepper. Pop the tortillas (3 per person) into the oven to warm through, 1-2 mins.\"\n- Step 6: \"When everything's ready, thinly slice the crispy chicken widthways into 2cm thick slices. Share the tortillas between your plates. Top with the salad, crispy chicken slices, a drizzle of buffalo sauce and a sprinkle of cheese - as much as you'd like. Serve your tacos and chips with the mayo for dipping.\"\n\n**5. Format as JSON:**\n- Ensure all required fields are present.\n- Convert quantities to numbers.\n- Ensure units are strings.",
                            "thought": True
                        },
                        {
                            "text": "{\n  \"title\": \"Crispy Buffalo Chicken Tacos and Chips\",\n  \"description\": \"with Cheddar and Honey Mustard Baby Gem Salad\",\n  \"servings\": 2,\n  \"ingredients\": [\n    {\n      \"name\": \"Potatoes\",\n      \"quantity\": 450,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Chicken Breast\",\n      \"quantity\": 2,\n      \"unit\": \"pieces\"\n    },\n    {\n      \"name\": \"Panko Breadcrumbs\",\n      \"quantity\": 50,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Central American Style Spice Mix\",\n      \"quantity\": 1,\n      \"unit\": \"sachet\"\n    },\n    {\n      \"name\": \"Mature Cheddar Cheese\",\n      \"quantity\": 40,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Baby Gem Lettuce\",\n      \"quantity\": 1,\n      \"unit\": \"piece\"\n    },\n    {\n      \"name\": \"Cider Vinegar\",\n      \"quantity\": 15,\n      \"unit\": \"ml\"\n    },\n    {\n      \"name\": \"Sriracha\",\n      \"quantity\": 30,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Honey\",\n      \"quantity\": 15,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Creme Fraiche\",\n      \"quantity\": 75,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Honey and Mustard Dressing\",\n      \"quantity\": 30,\n      \"unit\": \"g\"\n    },\n    {\n      \"name\": \"Plain Taco Tortilla\",\n      \"quantity\": 6,\n      \"unit\": \"pieces\"\n    },\n    {\n      \"name\": \"Egg\",\n      \"quantity\": 1,\n      \"unit\": \"piece\"\n    },\n    {\n      \"name\": \"Salt for the Breadcrumbs\",\n      \"quantity\": 0.25,\n      \"unit\": \"tsp\"\n    },\n    {\n      \"name\": \"Water for the Sauce\",\n      \"quantity\": 100,\n      \"unit\": \"ml\"\n    },\n    {\n      \"name\": \"Mayonnaise\",\n      \"quantity\": 2,\n      \"unit\": \"tbsp\"\n    }\n  ],\n  \"instructions\": [\n    {\n      \"step\": 1,\n      \"text\": \"Preheat your oven to 220°C/200°C fan/gas mark 7. Chop the potatoes lengthways into 1cm slices, then chop into 1cm wide chips (no need to peel). Pop the chips onto a large baking tray. Drizzle with oil, season with salt and pepper, then toss to coat. Spread out in a single layer. When the oven is hot, bake on the top shelf until golden, 25-30 mins. Turn halfway through.\"\n    },\n    {\n      \"step\": 2,\n      \"text\": \"Meanwhile, sandwich each chicken breast between two pieces of baking paper or cling film. Pop onto a board, then give it a bash with the bottom of a saucepan until it's 1-2 cm thick. Season with salt and pepper. Crack the egg into a medium bowl and whisk. In another medium bowl, combine the panko breadcrumbs, Central American style spice mix and the salt. Dip the chicken into the egg and then the breadcrumbs, ensuring it's completely coated. Transfer to a clean plate.\"\n    },\n    {\n      \"step\": 3,\n      \"text\": \"Pop a large frying pan on high heat and add enough oil to coat the bottom of the pan. Once hot, carefully lay the chicken into the pan, and fry until golden brown, 2-3 mins on each side. Adjust the heat if necessary. Transfer the chicken to a large baking tray, then bake on the middle shelf until cooked through, 8-10 mins.\"\n    },\n    {\n      \"step\": 4,\n      \"text\": \"While the chicken bakes, grate the Cheddar cheese. Trim the baby gem, halve lengthways, then thinly slice. Carefully discard the hot oil from the chicken pan, then wipe out and return to medium-high heat. Pour in the cider vinegar and water for the sauce. Bring to the boil, then simmer until reduced by half, 2-3 mins. Stir in the sriracha and honey until thickened, 1-2 mins.\"\n    },\n    {\n      \"step\": 5,\n      \"text\": \"Once your buffalo sauce has thickened, stir through the creme fraiche until piping hot, 1-2 mins. Season with salt and pepper. Add a splash of water if it's a little too thick, then remove from the heat. When the chips have 5 mins left, in a medium bowl, combine the baby gem with the honey and mustard dressing. Season with salt and pepper. Pop the tortillas (3 per person) into the oven to warm through, 1-2 mins.\"\n    },\n    {\n      \"step\": 6,\n      \"text\": \"When everything's ready, thinly slice the crispy chicken widthways into 2cm thick slices. Share the tortillas between your plates. Top with the salad, crispy chicken slices, a drizzle of buffalo sauce and a sprinkle of cheese - as much as you'd like. Serve your tacos and chips with the mayo for dipping.\"\n    }\n  ]\n}"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "citationMetadata": {
                    "citationSources": [
                        {
                            "startIndex": 2568,
                            "endIndex": 2774,
                            "uri": "https://the-messenger-online.com/perch/resources/messengeraug24.pdf",
                            "license": ""
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 909,
            "candidatesTokenCount": 1261,
            "totalTokenCount": 3115,
            "promptTokensDetails": [
                {
                    "modality": "TEXT",
                    "tokenCount": 393
                },
                {
                    "modality": "DOCUMENT",
                    "tokenCount": 516
                }
            ],
            "thoughtsTokenCount": 945,
            "serviceTier": "standard"
        },
        "modelVersion": "gemma-4-31b-it",
        "responseId": "EolMaoqdI62QnsEPkaeP2A4"
    }


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

    # Check for PDF content and return recipe response
    if _has_pdf_content(messages):
        _stats["total_image_requests"] += 1
        return JSONResponse(content=_get_recipe_response())

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

    # Check for PDF content and return recipe response
    if _has_pdf_content(messages):
        _stats["total_image_requests"] += 1
        return JSONResponse(content=_get_recipe_response())

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
