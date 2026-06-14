import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import logging

import database as db
from core.rate_limiter import init_rate_limiter
from core.router import init_router_service
from infrastructure.http_client import init_http_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
    init_router_service(http_client, rate_limiter)
    
    logger.info("Application services initialized")

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

# Mount static files for the Dashboard
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
