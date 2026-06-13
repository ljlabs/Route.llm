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
    """Get all logs."""
    try:
        return db.get_logs()
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


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