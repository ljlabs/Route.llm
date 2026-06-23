"""
Settings API Endpoints

Manage application settings like rate limiting and log retention.
"""

from fastapi import APIRouter, HTTPException, Request
from core.rate_limiter import get_rate_limiter, get_per_provider_limiter
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
            "rate_limit_tps": db.get_rate_limit_tps(),
            "max_tokens": db.get_max_tokens(),
            "response_format": db.get_response_format(),
            "disable_streaming": db.get_disable_streaming()
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

            # Update both the global rate limiter and per-provider global rate
            rate_limiter = get_rate_limiter()
            rate_limiter.set_rate(settings.rate_limit_tps)
            per_provider = get_per_provider_limiter()
            per_provider.set_global_rate(settings.rate_limit_tps)
            logger.info(f"Rate limit updated to {settings.rate_limit_tps} TPS")

        # Update max tokens if provided
        if settings.max_tokens is not None:
            db.set_max_tokens(settings.max_tokens)
            logger.info(f"Max tokens updated to {settings.max_tokens}")

        # Update response format if provided
        if settings.response_format is not None:
            db.set_response_format(settings.response_format)
            logger.info(f"Response format updated to {settings.response_format}")

        # Update disable_streaming if provided
        if settings.disable_streaming is not None:
            db.set_disable_streaming(settings.disable_streaming)
            logger.info(f"Disable streaming updated to {settings.disable_streaming}")

        return {"status": "success", "message": "Settings updated"}
    except ValueError as e:
        logger.error(f"Invalid settings value: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid value: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings")