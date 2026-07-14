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
async def list_models(request: Request):
    """List models using the OpenAI or Anthropic schema requested by the client."""
    try:
        mappings = db.get_model_mappings()
        model_ids = [mapping["model_id"] for mapping in mappings]

        # Add the active provider's model if it is not already mapped.
        active_provider = db.get_active_provider()
        if active_provider and active_provider["model_name"] not in model_ids:
            model_ids.append(active_provider["model_name"])

        # Anthropic and OpenAI share this path but use incompatible model schemas.
        # Anthropic clients identify their protocol with the required version header.
        if request.headers.get("anthropic-version"):
            models_data = [
                {
                    "type": "model",
                    "id": model_id,
                    "display_name": model_id,
                    "created_at": "1970-01-01T00:00:00Z",
                }
                for model_id in model_ids
            ]
            response = {"data": models_data, "has_more": False}
            if model_ids:
                response["first_id"] = model_ids[0]
                response["last_id"] = model_ids[-1]
            return response

        models_data = [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "router"}
            for model_id in model_ids
        ]
        return {"object": "list", "data": models_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models_no_prefix(request: Request):
    """Model list endpoint without the /v1/ prefix."""
    return await list_models(request)

@router.delete("/v1/models", status_code=status.HTTP_405_METHOD_NOT_ALLOWED)
async def delete_models_not_allowed():
    """DELETE /v1/models is not allowed."""
    raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Method not allowed")


@router.post("/v1/messages")
async def proxy_anthropic_messages(request: Request, anthropic_request: AnthropicRequest):
    """
    Anthropic-compatible messages endpoint.
    Routes to the active provider with necessary translation.
    """
    try:
        # Extract anthropic-version header if present
        anthropic_version = request.headers.get("anthropic-version", "2023-06-01")

        # Pydantic model converts to dict for the router service
        req_body = anthropic_request.model_dump(exclude_none=True)
        stream = anthropic_request.stream

        router_service = get_router_service()
        return await router_service.route_anthropic_request(req_body, stream=stream, anthropic_version=anthropic_version)

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
