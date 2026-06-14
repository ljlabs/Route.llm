"""
Metrics API Endpoints

Provide aggregated usage and performance statistics for the proxy.
"""

from fastapi import APIRouter, HTTPException
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

@router.get("")
async def get_metrics():
    """
    Get aggregated metrics summary per provider.
    Returns request counts, token usage, and average latency.
    """
    try:
        return db.get_metrics_summary()
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")

@router.get("/history")
async def get_latency_history(limit: int = 50):
    """
    Get recent latency history for all providers.
    """
    try:
        return db.get_latency_history(limit)
    except Exception as e:
        logger.error(f"Failed to get latency history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve latency history")
