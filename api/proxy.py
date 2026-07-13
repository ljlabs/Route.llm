"""
LLM Proxy API Endpoints

Anthropic and OpenAI compatible endpoints that route to configured providers.
"""

from typing import Any, Dict, Union
from fastapi import APIRouter, HTTPException, Request, status
from core.router import get_router_service
from models.request import AnthropicRequest, OpenAIRequest
import logging
import database as db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proxy"])


@router.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models list endpoint.
    Returns available models from the active provider and model mappings.
    """
    try:
        # Get model mappings
        mappings = db.get_model_mappings()
        model_ids = [m["model_id"] for m in mappings]

        # Add active provider's model if not already in mappings
        active_prov = db.get_active_provider()
        if active_prov and active_prov["model_name"] not in model_ids:
            model_ids.append(active_prov["model_name"])

        # Return OpenAI-compatible format
        models_data = [{"id": m, "object": "model", "created": 0, "owned_by": "router"} for m in model_ids]
        return {"object": "list", "data": models_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models_no_prefix():
    """
    OpenAI-compatible models list endpoint (no /v1/ prefix).
    Some clients like Android Studio call /models instead of /v1/models.
    """
    return await list_models()

@router.delete("/v1/models", status_code=status.HTTP_405_METHOD_NOT_ALLOWED)
async def delete_models_not_allowed():
    """DELETE /v1/models is not allowed."""
    raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Method not allowed")


@router.post("/v1/messages")
async def proxy_anthropic_messages(request: AnthropicRequest):
    """
    Anthropic-compatible messages endpoint.
    Routes to the active provider with necessary translation.
    """
    try:
        # Pydantic model converts to dict for the router service
        req_body = request.model_dump(exclude_none=True)
        stream = request.stream
        
        router_service = get_router_service()
        return await router_service.route_anthropic_request(req_body, stream=stream)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Anthropic proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions")
async def proxy_openai_completions(request: OpenAIRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Routes to the active provider with necessary translation.
    """
    try:
        # Pydantic model converts to dict for the router service
        req_body = request.model_dump(exclude_none=True)
        stream = request.stream

        router_service = get_router_service()
        return await router_service.route_openai_request(req_body, stream=stream)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/completions")
async def proxy_openai_completions_no_prefix(request: Request):
    """
    OpenAI-compatible chat completions endpoint (no /v1/ prefix).
    Some clients like Android Studio call /chat/completions instead of /v1/chat/completions.
    """
    body = await request.json()

    # Parse with Pydantic model for validation
    from models.request import OpenAIRequest
    try:
        parsed = OpenAIRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return await proxy_openai_completions(parsed)
