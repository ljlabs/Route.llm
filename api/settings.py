"""
Settings API Endpoints

Manage application settings like rate limiting and log retention.
"""

from fastapi import APIRouter, HTTPException, Request
from core.rate_limiter import get_rate_limiter
from models.request import SettingsRequest
from models.provider import SettingsResponse
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings():
    """Get current settings."""
    try:
        return {
            "log_limit": db.get_log_limit(),
            "rate_limit_tps": db.get_rate_limit_tps()
        }
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.post("")
async def set_settings(settings: SettingsRequest):
    """Update settings."""
    try:
        # Update log limit if provided
        if settings.log_limit is not None:
            db.set_log_limit(settings.log_limit)
            logger.info(f"Log limit updated to {settings.log_limit}")

        # Update rate limit if provided
        if settings.rate_limit_tps is not None:
            db.set_rate_limit_tps(settings.rate_limit_tps)

            # Update the global rate limiter
            rate_limiter = get_rate_limiter()
            rate_limiter.set_rate(settings.rate_limit_tps)
            logger.info(f"Rate limit updated to {settings.rate_limit_tps} TPS")

        return {"status": "success", "message": "Settings updated"}
    except ValueError as e:
        logger.error(f"Invalid settings value: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid value: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings")