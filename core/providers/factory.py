"""
Provider Factory

Creates provider instances based on configuration.
"""

from typing import Any, Dict, Optional
import logging

from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .mistral import MistralProvider
from .openrouter import OpenRouterProvider
from .embedding import EmbeddingProvider
from .nvidia_nim import NvidiaNimEmbeddingProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating provider instances based on API type."""
    
    # Mapping of API types to provider classes
    _PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "mistral": MistralProvider,
        "openrouter": OpenRouterProvider,
        "embedding": EmbeddingProvider,
        "embedding_nvidia_nim": NvidiaNimEmbeddingProvider,
    }
    
    @classmethod
    def create_provider(cls, config: Dict[str, Any]) -> BaseProvider:
        """
        Create a provider instance from configuration dictionary.
        
        Args:
            config: Dictionary containing:
                - name: Provider display name
                - api_type: Type of API ("openai", "anthropic", "gemini", "mistral", "openrouter")
                - endpoint_url: API endpoint URL
                - api_key: API key for authentication
                - model_name: Model identifier to use
                - is_active: Whether this is the active provider
                - id: (optional) Database ID
                
        Returns:
            BaseProvider instance
            
        Raises:
            ValueError: If api_type is unknown
        """
        api_type = config.get("api_type", "").lower()
        
        provider_class = cls._PROVIDER_MAP.get(api_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {api_type}. "
                           f"Supported types: {list(cls._PROVIDER_MAP.keys())}")
        
        # Extract provider properties from config
        provider_config = {
            "name": config.get("name", "Unknown"),
            "endpoint_url": config.get("endpoint_url", ""),
            "api_key": config.get("api_key", ""),
            "model_name": config.get("model_name", ""),
            "is_active": bool(config.get("is_active", 0)),
            "provider_id": config.get("id"),
            "rate_limit_tps": config.get("rate_limit_tps"),
            "max_tokens": config.get("max_tokens"),
        }
        
        logger.debug(f"Creating {api_type} provider: {provider_config['name']}")
        
        return provider_class(**provider_config)
    
    @classmethod
    def register_provider(cls, api_type: str, provider_class: type):
        """
        Register a new provider type.
        
        Args:
            api_type: String identifier for the provider type
            provider_class: Provider class to instantiate
        """
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"{provider_class} must extend BaseProvider")
        cls._PROVIDER_MAP[api_type.lower()] = provider_class
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported provider types."""
        return list(cls._PROVIDER_MAP.keys())