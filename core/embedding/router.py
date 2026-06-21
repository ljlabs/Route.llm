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

        start_time = time.perf_counter()

        try:
            response = await self.http_client.post(
                provider.endpoint_url,
                json=wrapped,
                headers=provider.get_headers()
            )
            logger.info(f"Request To Nvidia NIM: {wrapped}")
        except httpx.RequestError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/embeddings",
                request_body=req_body_str,
                response_status=502,
                response_body=str(e),
                latency_ms=latency_ms
            )
            raise HTTPException(status_code=502, detail=f"Backend unavailable: {e}")

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if response.status_code >= 400:
            error_text = response.text
            self._log_request(
                provider=provider,
                method="POST",
                path="/v1/embeddings",
                request_body=req_body_str,
                response_status=response.status_code,
                response_body=error_text,
                latency_ms=latency_ms
            )
            detail = f"Backend returned {response.status_code}"
            try:
                err = response.json()
                if "error" in err:
                    detail += f": {err['error'].get('message', '')}"
            except Exception:
                if error_text:
                    detail += f": {error_text[:200]}"
            raise HTTPException(status_code=response.status_code, detail=detail)

        response_json = response.json()

        # Extract token usage
        usage = response_json.get("usage", {})
        tokens_sent = usage.get("prompt_tokens", 0)
        tokens_received = usage.get("total_tokens", 0)

        self._log_request(
            provider=provider,
            method="POST",
            path="/v1/embeddings",
            request_body=req_body_str,
            response_status=200,
            response_body=json.dumps(response_json, indent=2)[:500],
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
            latency_ms=latency_ms
        )

        return JSONResponse(content=response_json, status_code=200)

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
            logger.error(f"Failed to log embedding request: {e}")


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
