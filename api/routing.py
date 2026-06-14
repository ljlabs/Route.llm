"""
Model Routing API Endpoints

Endpoints for managing the mapping between Model IDs and Providers.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
import database as db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/routing", tags=["routing"])


@router.get("", response_model=List[Dict[str, Any]])
async def get_mappings():
    """Get all model-to-provider mappings."""
    try:
        return db.get_model_mappings()
    except Exception as e:
        logger.error(f"Failed to get mappings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve routing mappings")


@router.post("")
async def add_mapping(mapping: Dict[str, Any]):
    """
    Add or update a model mapping.
    Expected body: {"model_id": "my-model", "provider_id": 1}
    """
    model_id = mapping.get("model_id")
    provider_id = mapping.get("provider_id")

    if not model_id or not provider_id:
        raise HTTPException(status_code=400, detail="model_id and provider_id are required")

    try:
        db.add_model_mapping(model_id, provider_id)
        return {"status": "success", "message": f"Model {model_id} mapped to provider {provider_id}"}
    except Exception as e:
        logger.error(f"Failed to add mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to create routing mapping")


@router.delete("/{model_id}")
async def delete_mapping(model_id: str):
    """Delete a model mapping."""
    try:
        db.delete_model_mapping(model_id)
        return {"status": "success", "message": f"Mapping for {model_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete routing mapping")
