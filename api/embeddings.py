"""
Embedding Proxy API Endpoint

OpenAI-compatible /v1/embeddings endpoint that routes to configured embedding providers.
"""

from fastapi import APIRouter, HTTPException
from models.request import EmbeddingRequest
from core.embedding.router import get_embedding_router_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embedding"])


@router.post("/v1/embeddings")
async def proxy_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    try:
        req_body = request.model_dump(exclude_none=True)
        router_service = get_embedding_router_service()
        return await router_service.route_embedding_request(req_body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embedding proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
