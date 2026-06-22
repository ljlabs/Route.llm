"""
Embedding Router Service

Routes embedding requests to the active embedding provider.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import httpx

from core.providers.service import get_provider_service
from core.providers.base import BaseProvider
from core.rate_limiter import PerProviderRateLimiter, get_per_provider_limiter

logger = logging.getLogger(__name__)


class EmbeddingRouterService:
    """Routes OpenAI-compatible embedding requests to active embedding providers."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        per_provider_limiter: PerProviderRateLimiter = None
    ):
        self.provider_service = get_provider_service()
        self.http_client = http_client
        self.per_provider_limiter = per_provider_limiter or get_per_provider_limiter()

    async def route_embedding_request(
        self,
        embedding_request: Dict[str, Any]
    ) -> JSONResponse:
        """Route an embedding request to the active embedding provider."""
        import database as db

        model_name = embedding_request.get("model")

        # Try model-specific routing first, then fall back to active embedding provider
        provider = None
        if model_name:
            provider = self.provider_service.get_provider_by_model(model_name)

        if not provider:
            provider = self.provider_service.get_active_embedding_provider()

        if not provider:
            raise HTTPException(
                status_code=400,
                detail="No active embedding provider configured"
            )

        # Rate limiting
        await self.per_provider_limiter.wait_for_provider(
            provider.provider_id, provider.rate_limit_tps
        )

        # Wrap and send
        logger.info(f"Provider Instance: {provider}")
        wrapped = provider.wrap_request(embedding_request)
        req_body_str = json.dumps(embedding_request, indent=2)
        provider_req_str = json.dumps(wrapped, indent=2)

        # Stage 1 — Router received
        request_id = db.start_request_log(
            provider_name=provider.name,
            request_method="POST",
            request_path="/v1/embeddings",
            request_body=req_body_str,
        )

        # Stage 2 — About to send to provider
        db.add_log_event(request_id, stage="provider_request", body=provider_req_str)

        start_time = time.perf_counter()

        try:
            response = await self.http_client.post(
                provider.endpoint_url,
                json=wrapped,
                headers=provider.get_headers()
            )
            logger.info(f"Request To Provider: {wrapped}")
        except httpx.RequestError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            # Stage 3 — Provider error (connection failed)
            db.add_log_event(request_id, stage="provider_response", body=str(e), status_code=502)
            # Stage 4 — Error response to client
            db.complete_request_log(
                request_id=request_id,
                response_status=502,
                response_body=str(e),
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=502, detail=f"Backend unavailable: {e}")

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Stage 3 — Response received from provider
        raw_response_text = response.text
        db.add_log_event(
            request_id,
            stage="provider_response",
            body=raw_response_text[:500],
            status_code=response.status_code,
        )

        if response.status_code >= 400:
            error_text = raw_response_text
            detail = f"Backend returned {response.status_code}"
            try:
                err = response.json()
                if "error" in err:
                    detail += f": {err['error'].get('message', '')}"
            except Exception:
                if error_text:
                    detail += f": {error_text[:200]}"
            # Stage 4 — Error response to client
            db.complete_request_log(
                request_id=request_id,
                response_status=response.status_code,
                response_body=error_text,
                latency_ms=latency_ms,
            )
            raise HTTPException(status_code=response.status_code, detail=detail)

        response_json = response.json()

        # Extract token usage
        usage = response_json.get("usage", {})
        tokens_sent = usage.get("prompt_tokens", 0)
        tokens_received = usage.get("total_tokens", 0)

        # Stage 4 — Successful response to client
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body=json.dumps(response_json, indent=2)[:500],
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            latency_ms=latency_ms,
        )

        return JSONResponse(content=response_json, status_code=200)


# Global instance
_embedding_router_service: Optional[EmbeddingRouterService] = None


def get_embedding_router_service() -> EmbeddingRouterService:
    global _embedding_router_service
    if _embedding_router_service is None:
        raise RuntimeError("EmbeddingRouterService not initialized.")
    return _embedding_router_service


def init_embedding_router_service(
    http_client: httpx.AsyncClient,
    per_provider_limiter: PerProviderRateLimiter = None
) -> EmbeddingRouterService:
    global _embedding_router_service
    _embedding_router_service = EmbeddingRouterService(http_client, per_provider_limiter)
    return _embedding_router_service
