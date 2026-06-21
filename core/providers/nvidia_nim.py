"""
Nvidia NIM Embedding Provider

Handles embedding requests for Nvidia NIM endpoints (e.g. nvidia/nv-embedcode-7b-v1).
Passes through input_type, encoding_format, and truncate fields required by NIM.
"""

from typing import Any, Dict
from .base import BaseProvider
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NvidiaNimEmbeddingProvider(BaseProvider):
    """Provider for Nvidia NIM embedding APIs."""

    def __init__(self, **kwargs):
        kwargs.pop("api_type", None)
        super().__init__(api_type="embedding_nvidia_nim", **kwargs)

    def wrap_request(self, embedding_request: Dict[str, Any]) -> Dict[str, Any]:
        """Build NIM embedding request, forwarding NIM-specific fields."""
        payload = {
            "model": self.model_name,
            "input": embedding_request.get("input", ""),
        }
        payload["input_type"] = embedding_request.get("input_type", "query")
        for key in ("encoding_format", "truncate"):
            if key in embedding_request:
                payload[key] = embedding_request[key]
        return payload

    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        return provider_response

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_stream_translator(self, target_format: str = "anthropic"):
        return None

    def requires_translation(self) -> bool:
        return False
