"""
Gemini Provider Implementation

Extends OpenAI provider with Gemini-specific sanitization.
"""

from typing import Any, Dict
from .openai import OpenAIProvider
from .translation import sanitize_openai_payload


class GeminiProvider(OpenAIProvider):
    """Provider for Google Gemini API (OpenAI-compatible format)."""
    
    def __init__(self, **kwargs):
        super().__init__(api_type="gemini", **kwargs)
    
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to Gemini format with strict sanitization."""
        # Use the parent translation
        wrapped = anthropic_to_openai_request(anthropic_request, self.model_name)
        # Apply Gemini-specific sanitization (allows extra_content for thought signature)
        return sanitize_openai_payload(wrapped, is_gemini=True)
    
    def sanitize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request specifically for Gemini API."""
        return sanitize_openai_payload(request, is_gemini=True)


# Import the translation function we need
from .translation import anthropic_to_openai_request