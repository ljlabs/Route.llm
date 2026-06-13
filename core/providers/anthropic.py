"""
Anthropic Provider Implementation

Pass-through provider for Anthropic API - no translation needed.
"""

from typing import Any, Dict
from .base import BaseProvider
from ..translation.stream_base import OpenAIToAnthropicStreamTranslator, PassthroughStreamTranslator


class AnthropicProvider(BaseProvider):
    """Provider for direct Anthropic API access."""
    
    def __init__(self, **kwargs):
        super().__init__(api_type="anthropic", **kwargs)
    
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """No translation needed - pass through with updated model name."""
        anthropic_request["model"] = self.model_name
        return anthropic_request
    
    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """No translation needed - return as-is."""
        return provider_response
    
    def get_headers(self) -> Dict[str, str]:
        """Get Anthropic-specific headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    
    def get_stream_translator(self, target_format: str = "anthropic"):
        """Get stream translator for this provider."""
        if target_format == "openai":
            return OpenAIToAnthropicStreamTranslator()
        return PassthroughStreamTranslator()
    
    def requires_translation(self) -> bool:
        """Anthropic doesn't require translation when talking to Anthropic API."""
        return False