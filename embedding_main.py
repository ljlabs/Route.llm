"""
Embedding Proxy Server

Separate FastAPI application that runs on port 8081.
Shares the same database and infrastructure as the main LLM router.
"""

import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

import database as db
from core.rate_limiter import init_rate_limiter, get_per_provider_limiter
from core.embedding.router import init_embedding_router_service
from infrastructure.http_client import init_http_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Support custom DB path for integration tests
_db_path = os.environ.get("EMBEDDING_DB_PATH")
if _db_path:
    db.DB_PATH = _db_path

app = FastAPI(title="Embedding Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    db.init_db()

    http_client = init_http_client()
    rate_limiter = init_rate_limiter()
    per_provider_limiter = get_per_provider_limiter()
    init_embedding_router_service(http_client, per_provider_limiter)

    logger.info("Embedding proxy services initialized")

    for route in app.routes:
        if hasattr(route, "path"):
            logger.info(f"  Route: {route.path} [{getattr(route, 'methods', 'mount')}]")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.debug(f"--> {request.method} {request.url.path} (query={request.url.query})")
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.debug(f"<-- {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms}ms)")
    return response


@app.on_event("shutdown")
async def shutdown_event():
    from infrastructure.http_client import get_http_client
    client = get_http_client()
    await client.close()
    logger.info("Embedding proxy shutdown complete")


# Include API Routers
from api.embeddings import router as embeddings_router
from api.providers import router as providers_router
from api.logs import router as logs_router

app.include_router(embeddings_router)
app.include_router(providers_router)
app.include_router(logs_router)

# Serve static files and embedding UI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pathlib

_static_dir = pathlib.Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return (_static_dir / "index.html").read_text(encoding="utf-8")

app.mount("/", StaticFiles(directory=str(_static_dir)), name="static")
