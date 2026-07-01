"""
Logs API Endpoints

View and manage request logs.
"""

from fastapi import APIRouter, HTTPException
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("")
async def get_logs():
    """Get all logs (with full lifecycle events)."""
    try:
        return db.get_logs()
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


@router.get("/metadata")
async def get_logs_metadata():
    """Get all logs metadata only (without lifecycle events) for fast initial load."""
    try:
        return db.get_logs_metadata()
    except Exception as e:
        logger.error(f"Failed to get logs metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs metadata")


@router.get("/{log_id}/events")
async def get_log_events(log_id: int):
    """Get lifecycle events for a specific log."""
    try:
        events = db.get_log_events(log_id)
        return events
    except Exception as e:
        logger.error(f"Failed to get log events for log {log_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve log events")


@router.delete("")
async def clear_logs():
    """Clear all logs."""
    try:
        db.clear_logs()
        logger.info("Logs cleared")
        return {"status": "success", "message": "Logs cleared"}
    except Exception as e:
        logger.error(f"Failed to clear logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear logs")