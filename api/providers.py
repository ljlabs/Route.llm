"""
Provider API Endpoints

CRUD operations for provider management.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Request
from models.provider import ProviderCreate, ProviderUpdate, ProviderResponse
from core.providers.service import get_provider_service
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])

# Get provider service
provider_service = get_provider_service()


@router.get("", response_model=List[ProviderResponse])
async def get_providers():
    """Get all providers."""
    try:
        providers = provider_service.get_all_providers()
        # Ensure is_active is boolean for Pydantic
        for p in providers:
            p["is_active"] = bool(p.get("is_active", 0))
        return providers
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve providers")


@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: int):
    """Get a specific provider by ID."""
    providers = provider_service.get_all_providers()
    
    for provider in providers:
        if provider.get("id") == provider_id:
            provider = provider.copy()
            provider["is_active"] = bool(provider.get("is_active", 0))
            return provider
    
    raise HTTPException(status_code=404, detail="Provider not found")


@router.post("")
async def add_provider(provider: ProviderCreate):
    """Add a new provider."""
    try:
        # Add provider
        provider_service.add_provider(provider.model_dump())
        return {"status": "success", "message": "Provider added"}
    except Exception as e:
        logger.error(f"Failed to add provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to add provider")


@router.put("/{provider_id}")
async def update_provider(provider_id: int, provider_update: ProviderUpdate):
    """Update a provider."""
    try:
        # Get existing provider
        providers = provider_service.get_all_providers()
        existing = None
        for p in providers:
            if p.get("id") == provider_id:
                existing = p
                break
        
        if not existing:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        # Merge existing with updates
        update_data = provider_update.model_dump(exclude_unset=True)
        merged_config = existing.copy()
        merged_config.update(update_data)
        
        # Update provider
        provider_service.update_provider(provider_id, merged_config)
        
        return {"status": "success", "message": "Provider updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to update provider")


@router.delete("/{provider_id}")
async def delete_provider(provider_id: int):
    """Delete a provider."""
    try:
        provider_service.delete_provider(provider_id)
        return {"status": "success", "message": "Provider deleted"}
    except Exception as e:
        logger.error(f"Failed to delete provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete provider")


@router.post("/{provider_id}/active")
async def activate_provider(provider_id: int):
    """Set a provider as active."""
    try:
        # Check if provider exists
        providers = provider_service.get_all_providers()
        found = False
        for p in providers:
            if p.get("id") == provider_id:
                found = True
                break
        
        if not found:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        provider_service.set_active_provider(provider_id)
        return {"status": "success", "message": "Provider activated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate provider")