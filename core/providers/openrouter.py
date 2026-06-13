"""
OpenRouter Provider Implementation

OpenRouter is OpenAI-compatible but has some specific requirements.
"""

from typing import Any, Dict
from .openai import OpenAIProvider
from .translation import sanitize_openai_payload, anthropic_to_openai_request


class OpenRouterProvider(OpenAIProvider):
    """Provider for OpenRouter.ai (OpenAI-compatible with extra headers)."""
    
    def __init__(self, **kwargs):
        super().__init__(api_type="openrouter", **kwargs)
    
    def get_headers(self) -> Dict[str, str]:
        """Get OpenRouter-specific headers including HTTP Referer."""
        headers = super().get_headers()
        # OpenRouter requires HTTP Referer for free tier
        headers["HTTP-Referer"] = "https://github.com/jordang7/model_router"
        headers["X-Title"] = "Model Router"
        return headers
    
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to OpenRouter format."""
        wrapped = anthropic_to_openai_request(anthropic_request, self.model_name)
        return sanitize_openai_payload(wrapped, is_gemini=False)