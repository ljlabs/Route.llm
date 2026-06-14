"""
Base Provider Abstract Class

All provider implementations must extend this class and implement the abstract methods.
Each provider encapsulates its own request wrapping, response unwrapping, and streaming logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        name: str,
        endpoint_url: str,
        api_key: str,
        model_name: str,
        api_type: str,
        is_active: bool = False,
        provider_id: Optional[int] = None,
        rate_limit_tps: Optional[float] = None
    ):
        self.name = name
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        self.api_type = api_type
        self.is_active = is_active
        self.provider_id = provider_id
        self.rate_limit_tps = rate_limit_tps
    
    @abstractmethod
    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert incoming Anthropic request to provider-specific format.
        
        Args:
            anthropic_request: Request in Anthropic /v1/messages format
            
        Returns:
            Request formatted for the provider's API
        """
        pass
    
    @abstractmethod
    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert provider response back to Anthropic format.
        
        Args:
            provider_response: Response from the provider's API
            
        Returns:
            Response in Anthropic /v1/messages format
        """
        pass
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """
        Get provider-specific HTTP headers.
        
        Returns:
            Dictionary of headers for API requests
        """
        pass
    
    @abstractmethod
    def get_stream_translator(self, target_format: str = "anthropic"):
        """
        Get the stream translator for converting streaming responses.
        
        Args:
            target_format: The target format ("anthropic" or "openai")
            
        Returns:
            StreamTranslator instance for handling streaming responses
        """
        pass
    
    @abstractmethod
    def requires_translation(self) -> bool:
        """
        Whether this provider requires translation between Anthropic and OpenAI formats.
        
        Returns:
            True if translation is required, False if formats are compatible
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} api_type={self.api_type}>"
    
    def to_config_dict(self) -> Dict[str, Any]:
        """Return provider configuration as dictionary for database storage."""
        return {
            "name": self.name,
            "api_type": self.api_type,
            "endpoint_url": self.endpoint_url,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "is_active": 1 if self.is_active else 0
        }