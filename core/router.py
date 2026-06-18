"""
Router Service

Main routing logic that coordinates providers, HTTP client, and rate limiting.
"""

import json
import logging
from typing import Any, Dict, Optional
import time

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from .providers.service import get_provider_service
from .providers.base import BaseProvider
from .rate_limiter import RateLimiter, PerProviderRateLimiter, get_per_provider_limiter

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
        per_provider_limiter: PerProviderRateLimiter = None,
        logger_service=None
    ):
        self.provider_service = get_provider_service()
        self.http_client = http_client
        self.rate_limiter = rate_limiter
        self.per_provider_limiter = per_provider_limiter or get_per_provider_limiter()
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

        # Apply rate limiting (per-provider if configured, else global)
        await self.per_provider_limiter.wait_for_provider(provider.provider_id, provider.rate_limit_tps)

        # Inject max_tokens if client didn't provide one
        if "max_tokens" not in anthropic_request or not anthropic_request.get("max_tokens"):
            import database as db
            effective_max_tokens = provider.max_tokens or db.get_max_tokens()
            anthropic_request["max_tokens"] = effective_max_tokens

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
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e.response, req_body_str, path="/v1/messages")
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/messages",
                request_body=req_body_str,
                response_status=500,
                response_body=str(e)
            )
            raise HTTPException(status_code=500, detail=str(e) or "Unknown routing error")

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

        # Apply rate limiting (per-provider if configured, else global)
        await self.per_provider_limiter.wait_for_provider(provider.provider_id, provider.rate_limit_tps)

        # Inject max_tokens if client didn't provide one
        if "max_tokens" not in openai_request or not openai_request.get("max_tokens"):
            import database as db
            effective_max_tokens = provider.max_tokens or db.get_max_tokens()
            openai_request["max_tokens"] = effective_max_tokens

        # For now, we translate OpenAI -> Anthropic then use provider.wrap_request
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
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e.response, req_body_str, path="/v1/chat/completions")
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/chat/completions",
                request_body=req_body_str,
                response_status=500,
                response_body=str(e)
            )
            raise HTTPException(status_code=500, detail=str(e) or "Unknown routing error")

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

        start_time = time.perf_counter()

        # Send request
        response = await self.http_client.post(
            provider.endpoint_url,
            json=wrapped_request,
            headers=provider.get_headers()
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

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

        # Extract token usage from raw response
        tokens_sent = 0
        tokens_received = 0
        usage = response_json.get("usage", {})
        if usage:
            # Support both OpenAI and Anthropic formats
            tokens_sent = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            tokens_received = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        # Fallback to estimation if usage is missing
        if tokens_sent == 0:
            tokens_sent = self._estimate_request_tokens(original_request)
        if tokens_received == 0:
            tokens_received = self._estimate_response_tokens(response_json)

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
            response_body=json.dumps(response_json, indent=2),
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            latency_ms=latency_ms
        )

        return JSONResponse(content=final_response, status_code=200)

    def _extract_usage_from_line(self, line: str) -> Dict[str, int]:
        """Attempt to extract token usage from a single SSE line."""
        usage = {"tokens_sent": 0, "tokens_received": 0}
        if not line.startswith("data:"):
            return usage

        try:
            data_content = line.replace("data:", "").strip()
            if not data_content or data_content == "[DONE]":
                return usage

            data = json.loads(data_content)

            # OpenAI format: usage in the choice or top level
            if "usage" in data:
                u = data["usage"]
                usage["tokens_sent"] = u.get("prompt_tokens") or 0
                usage["tokens_received"] = u.get("completion_tokens") or 0
            elif "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "usage" in choice:
                    u = choice["usage"]
                    usage["tokens_sent"] = u.get("prompt_tokens") or 0
                    usage["tokens_received"] = u.get("completion_tokens") or 0

            # Anthropic format: usage in message_delta
            elif data.get("type") == "message_delta":
                u = data.get("usage", {})
                usage["tokens_sent"] = u.get("input_tokens") or 0
                usage["tokens_received"] = u.get("output_tokens") or 0

        except Exception:
            pass

        return usage

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length (heuristic: 4 chars/token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _estimate_request_tokens(self, request_body: Any) -> int:
        """Estimate tokens in the request body."""
        try:
            data = request_body if isinstance(request_body, dict) else json.loads(request_body)
            text = ""
            for msg in data.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str): text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item.get("text", "")
            return self._estimate_tokens(text) if text else self._estimate_tokens(str(request_body))
        except Exception:
            return self._estimate_tokens(str(request_body))

    def _estimate_response_tokens(self, response_body: Any) -> int:
        """Estimate tokens in the response body."""
        try:
            data = response_body if not isinstance(response_body, str) else json.loads(response_body)
            text = ""
            if isinstance(data, list):
                for block in data:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
            elif isinstance(data, dict):
                if "choices" in data:
                    for choice in data["choices"]:
                        msg = choice.get("message", {})
                        text += msg.get("content") or msg.get("text") or ""
                else:
                    text += data.get("text") or data.get("content") or ""
            return self._estimate_tokens(text) if text else self._estimate_tokens(str(response_body))
        except Exception:
            return self._estimate_tokens(str(response_body))

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
            start_time = time.perf_counter()
            latency_ms = 0
            accumulated_blocks = []
            tokens_sent = 0
            tokens_received = 0
            first_byte_received = False

            try:
                async for line in stream_translator.translate_stream(
                    response,
                    provider_config,
                    accumulated_blocks
                ):
                    if not first_byte_received:
                        first_byte_received = True

                    # Extract usage from the line
                    u = self._extract_usage_from_line(line)
                    tokens_sent += u["tokens_sent"]
                    tokens_received += u["tokens_received"]

                    yield line
            finally:
                # Total latency: time to last token (completion)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                
                # Fallback to estimation if usage is missing
                if tokens_sent == 0:
                    tokens_sent = self._estimate_request_tokens(original_request)
                if tokens_received == 0:
                    tokens_received = self._estimate_response_tokens(accumulated_blocks)

                # Log the completed stream
                self._log_request(
                    provider=provider,
                    method="POST",
                    path=path,
                    request_body=req_body_str,
                    response_status=200,
                    response_body=json.dumps(accumulated_blocks, indent=2) if accumulated_blocks else "[Streamed response]",
                    tokens_sent=tokens_sent,
                    tokens_received=tokens_received,
                    latency_ms=latency_ms
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
        response_body: str,
        tokens_sent: int = 0,
        tokens_received: int = 0,
        latency_ms: int = 0
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
                response_body=response_body,
                tokens_sent=tokens_sent,
                tokens_received=tokens_received,
                latency_ms=latency_ms
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
    per_provider_limiter: PerProviderRateLimiter = None,
    logger_service=None
) -> RouterService:
    """Initialize the global router service."""
    global _router_service
    _router_service = RouterService(http_client, rate_limiter, per_provider_limiter, logger_service)
    return _router_service
