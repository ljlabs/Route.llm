"""
Mistral Provider Implementation

Extends OpenAI provider with Mistral-specific sanitization (stricter than OpenAI).
"""

from typing import Any, Dict
from .openai import OpenAIProvider
from .translation import sanitize_openai_payload, anthropic_to_openai_request


class MistralProvider(OpenAIProvider):
    """Provider for Mistral AI API (OpenAI-compatible format but stricter)."""
    
    def __init__(self, **kwargs):
        super().__init__(api_type="mistral", **kwargs)
    
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to Mistral format with strict sanitization."""
        # Use the parent translation
        wrapped = anthropic_to_openai_request(anthropic_request, self.model_name)
        # Apply strict sanitization (Mistral rejects unknown fields)
        return sanitize_openai_payload(wrapped, is_gemini=False)
    
    def sanitize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request specifically for Mistral API (very strict)."""
        return sanitize_openai_payload(request, is_gemini=False)