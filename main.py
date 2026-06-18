import os
import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import logging

import database as db
from core.rate_limiter import init_rate_limiter, get_per_provider_limiter
from core.router import init_router_service
from infrastructure.http_client import init_http_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Proxy & Router", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize database
    db.init_db()
    
    # Initialize infrastructure
    http_client = init_http_client()
    
    # Initialize core services
    rate_limiter = init_rate_limiter()
    per_provider_limiter = get_per_provider_limiter()
    init_router_service(http_client, rate_limiter, per_provider_limiter)
    
    logger.info("Application services initialized")

    # Log all registered routes
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
    logger.info("Application shutdown complete")

# Include API Routers
from api.providers import router as providers_router
from api.settings import router as settings_router
from api.logs import router as logs_router
from api.chat import router as chat_router
from api.proxy import router as proxy_router
from api.routing import router as routing_router
from api.metrics import router as metrics_router

app.include_router(providers_router)
app.include_router(settings_router)
app.include_router(logs_router)
app.include_router(chat_router)
app.include_router(proxy_router)
app.include_router(routing_router)
app.include_router(metrics_router)

# Serve the dashboard at root
from fastapi.responses import FileResponse

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Mount static assets (CSS, JS, images) — NOT at "/" to avoid catch-all
app.mount("/static", StaticFiles(directory=static_dir), name="static")
