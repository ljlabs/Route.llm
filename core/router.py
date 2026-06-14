"""
Router Service

Main routing logic that coordinates providers, HTTP client, and rate limiting.
"""

import json
import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from .providers.service import get_provider_service
from .providers.base import BaseProvider
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class RouterService:
    """
    Main service for routing requests to providers.
    
    Coordinates between provider service, HTTP client, and rate limiter.
    """
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        rate_limiter: RateLimiter,
        logger_service=None
    ):
        self.provider_service = get_provider_service()
        self.http_client = http_client
        self.rate_limiter = rate_limiter
        self.logger_service = logger_service
    
    async def route_anthropic_request(
        self,
        anthropic_request: Dict[str, Any],
        stream: bool = False
    ) -> JSONResponse | StreamingResponse:
        """
        Route an Anthropic-format request to the active provider.

        Args:
            anthropic_request: Request in Anthropic /v1/messages format
            stream: Whether to stream the response

        Returns:
            JSONResponse or StreamingResponse
        """
        # Extract model from request for model-based routing
        model_name = anthropic_request.get("model")

        # Try to get provider by model name first
        provider = None
        if model_name:
            provider = self.provider_service.get_provider_by_model(model_name)

        # If no provider found for the model, fall back to active provider
        if not provider:
            provider = self.provider_service.get_active_provider()

        if not provider:
            raise HTTPException(
                status_code=400,
                detail="No active provider configured"
            )

        # Apply rate limiting (use provider-specific limit if set, else global)
        await self.rate_limiter.wait(tps_override=provider.rate_limit_tps)

        # Wrap request to provider format
        wrapped_request = provider.wrap_request(anthropic_request)
        
        req_body_str = json.dumps(anthropic_request, indent=2)
        
        try:
            if stream:
                return await self._handle_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=anthropic_request,
                    req_body_str=req_body_str,
                    path="/v1/messages",
                    target_format="anthropic"
                )
            else:
                return await self._handle_non_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=anthropic_request,
                    req_body_str=req_body_str,
                    path="/v1/messages"
                )
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e, req_body_str, path="/v1/messages")
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def route_openai_request(
        self,
        openai_request: Dict[str, Any],
        stream: bool = False
    ) -> JSONResponse | StreamingResponse:
        """
        Route an OpenAI-format request to the active provider.

        Args:
            openai_request: Request in OpenAI /v1/chat/completions format
            stream: Whether to stream the response

        Returns:
            JSONResponse or StreamingResponse
        """
        # Extract model from request for model-based routing
        model_name = openai_request.get("model")

        # Try to get provider by model name first
        provider = None
        if model_name:
            provider = self.provider_service.get_provider_by_model(model_name)

        # If no provider found for the model, fall back to active provider
        if not provider:
            provider = self.provider_service.get_active_provider()

        if not provider:
            raise HTTPException(
                status_code=400,
                detail="No active provider configured"
            )
            
        # Apply rate limiting
        await self.rate_limiter.wait()
        
        # If backend is OpenAI-compatible and target is OpenAI, we might still need to 
        # translate if we want to support things like Anthropic -> OpenAI translation
        # but here we are starting from OpenAI format.
        
        # For now, we translate OpenAI -> Anthropic then use provider.wrap_request
        # This is a bit inefficient but ensures consistency if the provider is Anthropic.
        # If the provider is OpenAI, wrap_request will translate back to OpenAI.
        
        from .providers.translation import openai_to_anthropic_request
        anthropic_request = openai_to_anthropic_request(openai_request, provider.model_name)
        
        # Wrap request to provider format
        wrapped_request = provider.wrap_request(anthropic_request)
        
        req_body_str = json.dumps(openai_request, indent=2)
        
        try:
            if stream:
                return await self._handle_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=openai_request,
                    req_body_str=req_body_str,
                    path="/v1/chat/completions",
                    target_format="openai"
                )
            else:
                return await self._handle_non_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=openai_request,
                    req_body_str=req_body_str,
                    path="/v1/chat/completions",
                    is_openai_target=True
                )
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e, req_body_str, path="/v1/chat/completions")
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_non_streaming(
        self,
        provider: BaseProvider,
        wrapped_request: Dict[str, Any],
        original_request: Dict[str, Any],
        req_body_str: str,
        path: str = "/v1/messages",
        is_openai_target: bool = False
    ) -> JSONResponse:
        """Handle non-streaming request."""
        logger.info(f"Routing non-streaming request to {provider.name}")
        
        # Send request
        response = await self.http_client.post(
            provider.endpoint_url,
            json=wrapped_request,
            headers=provider.get_headers()
        )
        
        # Handle error responses
        if response.status_code >= 400:
            await self._handle_http_error(provider, response, req_body_str, path=path)
        
        # Get response data
        try:
            response_json = response.json()
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response: {str(e)}"
            )
        
        # Unwrap response to Anthropic format first
        anthropic_response = provider.unwrap_response(response_json)
        
        # If target was OpenAI, translate back to OpenAI
        if is_openai_target:
            from .providers.translation import anthropic_to_openai_response
            final_response = anthropic_to_openai_response(anthropic_response)
        else:
            final_response = anthropic_response
        
        # Log the request
        self._log_request(
            provider=provider,
            method="POST",
            path=path,
            request_body=req_body_str,
            response_status=response.status_code,
            response_body=json.dumps(response_json, indent=2)
        )
        
        return JSONResponse(content=final_response, status_code=200)
    
    async def _handle_streaming(
        self,
        provider: BaseProvider,
        wrapped_request: Dict[str, Any],
        original_request: Dict[str, Any],
        req_body_str: str,
        path: str = "/v1/messages",
        target_format: str = "anthropic"
    ) -> StreamingResponse:
        """Handle streaming request."""
        logger.info(f"Routing streaming request to {provider.name}")
        
        # Build streaming request
        request = self.http_client.build_request(
            "POST",
            provider.endpoint_url,
            json=wrapped_request,
            headers=provider.get_headers()
        )
        
        # Send request with streaming
        response = await self.http_client.send(request, stream=True)
        
        # Handle error responses
        if response.status_code >= 400:
            await response.aread()
            error_text = response.text
            await response.aclose()
            await self._handle_http_error(provider, response, req_body_str, error_text, path=path)
        
        # Get stream translator from provider
        stream_translator = provider.get_stream_translator(target_format)
        
        # Provider config for the translator
        provider_config = {
            "name": provider.name,
            "api_type": provider.api_type,
            "model_name": provider.model_name,
        }
        
        # Create async generator for streaming
        async def stream_generator():
            accumulated_blocks = []
            
            try:
                async for line in stream_translator.translate_stream(
                    response,
                    provider_config,
                    accumulated_blocks
                ):
                    yield line
            finally:
                # Log the completed stream
                self._log_request(
                    provider=provider,
                    method="POST",
                    path=path,
                    request_body=req_body_str,
                    response_status=200,
                    response_body=json.dumps(accumulated_blocks, indent=2) if accumulated_blocks else "[Streamed response]"
                )
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    
    async def _handle_http_error(
        self,
        provider: BaseProvider,
        response: httpx.Response,
        req_body_str: str,
        error_text: str = None,
        path: str = "/v1/messages"
    ):
        """Handle HTTP error responses."""
        status = response.status_code
        
        if error_text is None:
            try:
                error_text = response.text
            except Exception:
                error_text = ""
        
        # Try to extract error message
        error_detail = f"Backend returned {status}"
        try:
            error_json = response.json() if hasattr(response, 'json') else {}
            if "error" in error_json:
                error_detail += f": {error_json['error'].get('message', '')}"
            elif "detail" in error_json:
                error_detail += f": {error_json['detail']}"
        except Exception:
            if error_text:
                error_detail += f": {error_text[:200]}"
        
        # Add hints for common status codes
        if status == 401:
            error_detail = f"Unauthorized (401): Check your API Key. {error_detail}"
        elif status == 404:
            error_detail = f"Not Found (404): Check your Endpoint URL or Model Name. {error_detail}"
        elif status == 403:
            error_detail = f"Forbidden (403): Access denied. {error_detail}"
        elif status == 429:
            error_detail = f"Rate Limited (429): Too many requests. {error_detail}"
        
        # Log the error
        self._log_request(
            provider=provider,
            method="POST",
            path="/v1/messages",
            request_body=req_body_str,
            response_status=status,
            response_body=error_text or str(error_detail)
        )
        
        raise HTTPException(status_code=status, detail=error_detail)
    
    def _log_request(
        self,
        provider: BaseProvider,
        method: str,
        path: str,
        request_body: str,
        response_status: int,
        response_body: str
    ):
        """Log the request to database."""
        try:
            import database as db
            db.add_log(
                provider_name=provider.name,
                request_method=method,
                request_path=path,
                request_body=request_body,
                response_status=response_status,
                response_body=response_body
            )
        except Exception as e:
            logger.error(f"Failed to log request: {e}")


# Global service instance
_router_service: Optional[RouterService] = None


def get_router_service() -> RouterService:
    """Get the global router service instance."""
    global _router_service
    if _router_service is None:
        raise RuntimeError("RouterService not initialized. Call init_router_service first.")
    return _router_service


def init_router_service(
    http_client: httpx.AsyncClient,
    rate_limiter: RateLimiter,
    logger_service=None
) -> RouterService:
    """Initialize the global router service."""
    global _router_service
    _router_service = RouterService(http_client, rate_limiter, logger_service)
    return _router_service