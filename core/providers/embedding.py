"""
Embedding Provider Implementation

Handles embedding API requests (e.g. Google AI Studio gemini-embedding-2).
"""

from typing import Any, Dict
from .base import BaseProvider


class EmbeddingProvider(BaseProvider):
    """Provider for embedding APIs (OpenAI-compatible /v1/embeddings)."""

    def __init__(self, **kwargs):
        kwargs.pop("api_type", None)
        super().__init__(api_type="embedding", **kwargs)

    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Accept an OpenAI-compatible embedding request with 'model' and 'input'."""
        return {
            "model": self.model_name,
            "input": anthropic_request.get("input", "")
        }

    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """Embedding responses are already in OpenAI format -- pass through."""
        return provider_response

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_stream_translator(self, target_format: str = "anthropic"):
        """Embeddings never stream."""
        return None

    def requires_translation(self) -> bool:
        return False
