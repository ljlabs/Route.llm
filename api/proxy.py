"""
LLM Proxy API Endpoints

Anthropic and OpenAI compatible endpoints that route to configured providers.
"""

from typing import Any, Dict, Union
import re
from fastapi import APIRouter, HTTPException, Request, status
from core.router import get_router_service
from models.request import AnthropicRequest, OpenAIRequest
import logging
import database as db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proxy"])


_STANDARD_ANTHROPIC_MODEL_ALIAS = re.compile(
    r"^claude-(?:(?:3(?:-\d+)?)-(?:opus|sonnet|haiku)|(?:opus|sonnet|haiku)-4(?:-\d+)?)(?:-(?:\d{8}|latest))?$"
)


def _model_is_available(model_id: str) -> bool:
    """Return whether a model is explicitly advertised by this router."""
    if any(mapping["model_id"] == model_id for mapping in db.get_model_mappings()):
        return True
    active_provider = db.get_active_provider()
    return bool(active_provider and active_provider["model_name"] == model_id)


def _anthropic_model_is_available(model_id: str) -> bool:
    """Allow explicit models and standard Claude aliases with an active fallback."""
    if _model_is_available(model_id):
        return True
    return bool(
        db.get_active_provider()
        and _STANDARD_ANTHROPIC_MODEL_ALIAS.fullmatch(model_id)
    )


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


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    """Retrieve one OpenAI model object, or a compatibility-format 404."""
    mappings = db.get_model_mappings()
    model_ids = {mapping["model_id"] for mapping in mappings}
    active_provider = db.get_active_provider()
    if active_provider:
        model_ids.add(active_provider["model_name"])
    if model_id not in model_ids:
        raise HTTPException(status_code=404, detail=f"The model '{model_id}' does not exist")
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "router"}


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
        if not _anthropic_model_is_available(anthropic_request.model):
            raise HTTPException(status_code=404, detail=f"The model '{anthropic_request.model}' does not exist")

        router_service = get_router_service()
        return await router_service.route_anthropic_request(req_body, stream=stream, anthropic_version=anthropic_version)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Anthropic proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/messages/count_tokens")
async def count_anthropic_tokens(request: Request):
    """Provide a deterministic local estimate for Anthropic's optional token count API."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON request body")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    def content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    text = " ".join(content_text(message.get("content")) for message in messages if isinstance(message, dict))
    return {"input_tokens": max(1, len(text.split()))}


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
        if not _model_is_available(request.model):
            raise HTTPException(status_code=404, detail=f"The model '{request.model}' does not exist")

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
