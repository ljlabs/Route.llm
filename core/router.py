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

import database as db

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_logger = logging.getLogger("streaming")


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
        
        # Validate provider type: chat requests cannot go to embedding providers
        if provider.api_type and "embedding" in provider.api_type.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot route chat request to embedding provider '{provider.name}'. Please configure a chat provider as active."
            )

        req_body_str = json.dumps(anthropic_request, indent=2)

        # LOGGING Stage 1 — log as soon as the request reaches the router
        request_id = db.start_request_log(
            provider_name=provider.name,
            request_method="POST",
            request_path="/v1/messages",
            request_body=req_body_str,
        )

        # This is done after the first db.start_request_log so we can get the raw client request
        # Check if streaming is globally disabled
        if db.get_disable_streaming():
            stream = False
            anthropic_request["stream"] = False  # Also override in request body

        # Apply rate limiting (per-provider if configured, else global)
        await self.per_provider_limiter.wait_for_provider(provider.provider_id, provider.rate_limit_tps)

        # Inject max_tokens if client didn't provide one
        if "max_tokens" not in anthropic_request or not anthropic_request.get("max_tokens"):
            effective_max_tokens = provider.max_tokens or db.get_max_tokens()
            anthropic_request["max_tokens"] = effective_max_tokens

        # Wrap request to provider format
        logger.info("[ROUTER] wrapping request")
        wrapped_request = provider.wrap_request(anthropic_request)

        try:
            if stream:
                return await self._handle_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=anthropic_request,
                    req_body_str=req_body_str,
                    path="/v1/messages",
                    target_format="anthropic",
                    request_id=request_id,
                )
            else:
                return await self._handle_non_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=anthropic_request,
                    req_body_str=req_body_str,
                    path="/v1/messages",
                    request_id=request_id,
                    anthropic_request=anthropic_request,
                )
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e.response, req_body_str, path="/v1/messages", request_id=request_id)
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/messages",
                request_body=req_body_str,
                response_status=500,
                response_body=str(e),
                request_id=request_id,
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
        import database as db

        # Check if streaming is globally disabled
        if db.get_disable_streaming():
            stream = False
            openai_request["stream"] = False  # Also override in request body

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
        
        # Validate provider type: chat requests cannot go to embedding providers
        if provider.api_type and "embedding" in provider.api_type.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot route chat request to embedding provider '{provider.name}'. Please configure a chat provider as active."
            )

        req_body_str = json.dumps(openai_request, indent=2)

        # Stage 1 — log as soon as the request reaches the router
        request_id = db.start_request_log(
            provider_name=provider.name,
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body=req_body_str,
        )

        # Apply rate limiting (per-provider if configured, else global)
        await self.per_provider_limiter.wait_for_provider(provider.provider_id, provider.rate_limit_tps)

        # Inject max_tokens if client didn't provide one
        if "max_tokens" not in openai_request or not openai_request.get("max_tokens"):
            effective_max_tokens = provider.max_tokens or db.get_max_tokens()
            openai_request["max_tokens"] = effective_max_tokens

        # For now, we translate OpenAI -> Anthropic then use provider.wrap_request
        from .providers.translation import openai_to_anthropic_request
        anthropic_request = openai_to_anthropic_request(openai_request, provider.model_name)

        # Wrap request to provider format
        wrapped_request = provider.wrap_request(anthropic_request)

        try:
            if stream:
                return await self._handle_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=openai_request,
                    req_body_str=req_body_str,
                    path="/v1/chat/completions",
                    target_format="openai",
                    request_id=request_id,
                )
            else:
                return await self._handle_non_streaming(
                    provider=provider,
                    wrapped_request=wrapped_request,
                    original_request=openai_request,
                    req_body_str=req_body_str,
                    path="/v1/chat/completions",
                    is_openai_target=True,
                    request_id=request_id,
                    anthropic_request=anthropic_request,
                )
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            await self._handle_http_error(provider, e.response, req_body_str, path="/v1/chat/completions", request_id=request_id)
            raise
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/chat/completions",
                request_body=req_body_str,
                response_status=500,
                response_body=str(e),
                request_id=request_id,
            )
            raise HTTPException(status_code=500, detail=str(e) or "Unknown routing error")

    async def _handle_non_streaming(
        self,
        provider: BaseProvider,
        wrapped_request: Dict[str, Any],
        original_request: Dict[str, Any],
        req_body_str: str,
        path: str = "/v1/messages",
        is_openai_target: bool = False,
        request_id: Optional[int] = None,
        anthropic_request: Optional[Dict[str, Any]] = None,
    ) -> JSONResponse:
        """Handle non-streaming request."""
        import database as db
        logger.info(f"Routing non-streaming request to {provider.name}")

        # Check if provider supports native PDF and request contains PDFs
        if (anthropic_request is not None
                and getattr(provider, 'supports_native_pdf', False)
                and provider.detect_pdf_content(anthropic_request)):
            return await self._handle_native_pdf_request(
                provider=provider,
                anthropic_request=anthropic_request,
                original_request=original_request,
                req_body_str=req_body_str,
                path=path,
                is_openai_target=is_openai_target,
                request_id=request_id,
            )

        # Stage 2 — about to send to provider
        provider_req_body = json.dumps(wrapped_request, indent=2)
        db.add_log_event(request_id, stage="provider_request", body=provider_req_body)

        start_time = time.perf_counter()

        # Send request
        response = await self.http_client.post(
            provider.endpoint_url,
            json=wrapped_request,
            headers=provider.get_headers()
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Stage 3 — response received from provider
        raw_response_text = response.text
        db.add_log_event(
            request_id,
            stage="provider_response",
            body=raw_response_text,
            status_code=response.status_code,
        )

        # Handle error responses
        if response.status_code >= 400:
            await self._handle_http_error(provider, response, req_body_str, path=path, request_id=request_id)

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

        final_response_str = json.dumps(response_json, indent=2)

        # Stage 4 — finalise the log row (also records client_response event)
        db.complete_request_log(
            request_id=request_id,
            response_status=response.status_code,
            response_body=final_response_str,
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            latency_ms=latency_ms,
        )

        return JSONResponse(content=final_response, status_code=200)

    async def _handle_native_pdf_request(
        self,
        provider,
        anthropic_request: Dict[str, Any],
        original_request: Dict[str, Any],
        req_body_str: str,
        path: str = "/v1/messages",
        is_openai_target: bool = False,
        request_id: Optional[int] = None,
    ) -> JSONResponse:
        """Handle a request containing PDFs via Gemini's native generateContent endpoint."""
        import database as db
        logger.info(f"Routing native PDF request to {provider.name} via generateContent")

        native_request = provider.build_native_pdf_request(anthropic_request)
        native_endpoint = provider.build_native_pdf_endpoint()
        native_headers = {
            "x-goog-api-key": provider.api_key,
            "Content-Type": "application/json"
        }

        # Stage 2 — log the native request
        native_req_body = json.dumps(native_request, indent=2)
        db.add_log_event(request_id, stage="provider_request", body=native_req_body)

        start_time = time.perf_counter()

        response = await self.http_client.post(
            native_endpoint,
            json=native_request,
            headers=native_headers
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Stage 3 — response received
        raw_response_text = response.text
        db.add_log_event(
            request_id,
            stage="provider_response",
            body=raw_response_text,
            status_code=response.status_code,
        )

        if response.status_code >= 400:
            await self._handle_http_error(provider, response, req_body_str, path=path, request_id=request_id)

        try:
            gemini_response = response.json()
        except Exception as e:
            logger.error(f"Failed to parse native PDF response: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")

        # Translate Gemini native response to Anthropic format
        anthropic_response = provider.translate_native_response(gemini_response)

        # Extract token usage
        tokens_sent = anthropic_response.get("usage", {}).get("input_tokens", 0)
        tokens_received = anthropic_response.get("usage", {}).get("output_tokens", 0)
        if tokens_sent == 0:
            tokens_sent = self._estimate_request_tokens(original_request)
        if tokens_received == 0:
            tokens_received = self._estimate_response_tokens(gemini_response)

        # If target was OpenAI, translate back to OpenAI
        if is_openai_target:
            from .providers.translation import anthropic_to_openai_response
            final_response = anthropic_to_openai_response(anthropic_response)
        else:
            final_response = anthropic_response

        final_response_str = json.dumps(gemini_response, indent=2)

        # Stage 4 — finalise log
        db.complete_request_log(
            request_id=request_id,
            response_status=response.status_code,
            response_body=final_response_str,
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            latency_ms=latency_ms,
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
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text += item.get("text", "")
                            elif item.get("type") == "image_url":
                                # Estimate ~85 tokens per image (OpenAI pricing)
                                text += "x" * 340
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
        target_format: str = "anthropic",
        request_id: Optional[int] = None,
    ) -> StreamingResponse:
        """Handle streaming request."""
        import database as db
        logger.info(f"Routing streaming request to {provider.name}")

        # LOGGING Stage 2 — about to send to provider
        provider_req_body = json.dumps(wrapped_request, indent=2)
        db.add_log_event(request_id, stage="provider_request", body=provider_req_body)

        # Build streaming request
        request = self.http_client.build_request(
            "POST",
            provider.endpoint_url,
            json=wrapped_request,
            headers=provider.get_headers()
        )

        # Send request with streaming
        response = await self.http_client.send(request, stream=True)

        # LOGGING Stage 3 — first bytes received from provider (headers/status)
        db.add_log_event(
            request_id,
            stage="provider_response",
            body="[Streaming — body accumulating]",
            status_code=response.status_code,
        )

        # Handle error responses
        if response.status_code >= 400:
            await response.aread()
            error_text = response.text
            await response.aclose()
            await self._handle_http_error(provider, response, req_body_str, error_text, path=path, request_id=request_id)

        # Get stream translator from provider
        response_format = db.get_response_format()
        stream_translator = provider.get_stream_translator(target_format, validate_format=response_format)

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
            chunk_count = 0

            try:
                async for line in stream_translator.translate_stream(
                    response,
                    provider_config,
                    accumulated_blocks
                ):
                    chunk_count += 1
                    if not first_byte_received:
                        first_byte_received = True

                    # Extract usage from the line
                    u = self._extract_usage_from_line(line)
                    tokens_sent += u["tokens_sent"]
                    tokens_received += u["tokens_received"]

                    # Log the chunk if verbose streaming is enabled
                    stream_logger.debug(f"[CHUNK {chunk_count}] Response → Client: {line[:200]}...")
                    stream_logger.debug(f"[CHUNK {chunk_count}] Full: {line}")

                    yield line
            finally:
                # Total latency: time to last token (completion)
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                
                # Log summary
                stream_logger.debug(f"[STREAM COMPLETE] Total chunks: {chunk_count}, Latency: {latency_ms}ms")
                
                # Fallback to estimation if usage is missing
                if tokens_sent == 0:
                    tokens_sent = self._estimate_request_tokens(original_request)
                if tokens_received == 0:
                    tokens_received = self._estimate_response_tokens(accumulated_blocks)

                final_body = json.dumps(accumulated_blocks, indent=2) if accumulated_blocks else "[Streamed response]"

                # Log SSE validation results
                if hasattr(stream_translator, 'validation_checked') and stream_translator.validation_checked > 0:
                    validation_body = json.dumps({
                        "format": response_format,
                        "events_checked": stream_translator.validation_checked,
                        "warnings": stream_translator.validation_warnings,
                        "status": "pass" if not stream_translator.validation_warnings else "fail"
                    })
                    db.add_log_event(
                        request_id,
                        stage="sse_validation",
                        body=validation_body,
                        status_code=200 if not stream_translator.validation_warnings else 422,
                    )
                    if stream_translator.validation_warnings:
                        logger.warning(
                            "SSE_VALIDATION: %d warnings for %s format on %s",
                            len(stream_translator.validation_warnings),
                            response_format,
                            path,
                        )

                # Stage 4 — finalise log row (records client_response event)
                db.complete_request_log(
                    request_id=request_id,
                    response_status=200,
                    response_body=final_body,
                    tokens_sent=tokens_sent,
                    tokens_received=tokens_received,
                    latency_ms=latency_ms,
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
        path: str = "/v1/messages",
        request_id: Optional[int] = None,
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
            response_body=error_text or str(error_detail),
            request_id=request_id,
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
        latency_ms: int = 0,
        request_id: Optional[int] = None,
    ):
        """Log the request to database.

        If request_id is provided, complete the existing log row.
        Otherwise fall back to add_log (legacy path, e.g. error cases
        that fire before start_request_log was called).
        """
        try:
            import database as db
            if request_id is not None:
                db.complete_request_log(
                    request_id=request_id,
                    response_status=response_status,
                    response_body=response_body,
                    tokens_sent=tokens_sent,
                    tokens_received=tokens_received,
                    latency_ms=latency_ms,
                )
            else:
                db.add_log(
                    provider_name=provider.name,
                    request_method=method,
                    request_path=path,
                    request_body=request_body,
                    response_status=response_status,
                    response_body=response_body,
                    tokens_sent=tokens_sent,
                    tokens_received=tokens_received,
                    latency_ms=latency_ms,
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
