"""
LLM Proxy API Endpoints

Anthropic and OpenAI compatible endpoints that route to configured providers.
"""

from typing import Any, Dict, Union
from fastapi import APIRouter, HTTPException, Request
from core.router import get_router_service
from models.request import AnthropicRequest, OpenAIRequest
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proxy"])


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
