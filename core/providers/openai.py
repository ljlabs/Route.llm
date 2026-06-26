"""
OpenAI Provider Implementation

Translates between Anthropic and OpenAI request/response formats.
"""

from typing import Any, Dict
from .base import BaseProvider
from .translation import anthropic_to_openai_request, openai_to_anthropic_response, sanitize_openai_payload
from ..translation.stream_base import AnthropicToOpenAIStreamTranslator, PassthroughStreamTranslator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, OpenRouter, etc.)"""
    
    def __init__(self, api_type: str = "openai", **kwargs):
        super().__init__(api_type=api_type, **kwargs)
    
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to OpenAI format with sanitization."""
        logger.info("[OPEN AI] wrapping request")
        wrapped = anthropic_to_openai_request(anthropic_request, self.model_name)
        return sanitize_openai_payload(wrapped, is_gemini=False)
    
    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI response back to Anthropic format."""
        return openai_to_anthropic_response(provider_response)
    
    def get_headers(self) -> Dict[str, str]:
        """Get OpenAI-compatible headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_stream_translator(self, target_format: str = "anthropic", validate_format: str = None):
        """Get stream translator for this provider."""
        if target_format == "anthropic":
            return AnthropicToOpenAIStreamTranslator(validate_format=validate_format)
        return PassthroughStreamTranslator(validate_format=validate_format)
    
    def requires_translation(self) -> bool:
        """OpenAI requires translation to/from Anthropic format."""
        return True