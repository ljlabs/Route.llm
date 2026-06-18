"""
Provider Service

Manages provider instances and provides CRUD operations.
"""

from typing import Any, Dict, List, Optional
import logging
import database as db

from .base import BaseProvider
from .factory import ProviderFactory

logger = logging.getLogger(__name__)


class ProviderService:
    """
    Service for managing providers.
    
    Uses ProviderFactory to create provider instances from database configuration.
    """
    
    def __init__(self):
        self._factory = ProviderFactory()
        self._active_provider_cache: Optional[BaseProvider] = None
        self._cache_valid = False
    
    def _invalidate_cache(self):
        """Invalidate the provider cache."""
        self._active_provider_cache = None
        self._cache_valid = False
    
    def get_active_provider(self) -> Optional[BaseProvider]:
        """
        Get the currently active provider instance.
        
        Returns:
            BaseProvider instance or None if no active provider
        """
        if self._cache_valid and self._active_provider_cache:
            return self._active_provider_cache
        
        # Get provider config from database
        provider_config = db.get_active_provider()
        
        if not provider_config:
            return None
        
        # Create provider instance from config
        try:
            self._active_provider_cache = self._factory.create_provider(provider_config)
            self._cache_valid = True
            logger.info(f"Loaded active provider: {self._active_provider_cache}")
            return self._active_provider_cache
        except Exception as e:
            logger.error(f"Failed to create provider from config: {e}")
            return None
    
    def get_provider_by_id(self, provider_id: int) -> Optional[BaseProvider]:
        """
        Get a provider by its ID.

        Args:
            provider_id: Database ID of the provider

        Returns:
            BaseProvider instance or None if not found
        """
        providers = db.get_providers()

        for provider_config in providers:
            if provider_config.get("id") == provider_id:
                try:
                    return self._factory.create_provider(provider_config)
                except Exception as e:
                    logger.error(f"Failed to create provider {provider_id}: {e}")
                    return None

        return None

    def get_provider_by_model(self, model_name: str) -> Optional[BaseProvider]:
        """
        Get a provider based on the model mapping.

        Args:
            model_name: The model ID from the request

        Returns:
            BaseProvider instance if a mapping exists, None otherwise
        """
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT provider_id FROM model_mappings WHERE model_id = ?", (model_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self.get_provider_by_id(row["provider_id"])

        return None
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        Get all provider configurations from the database.
        
        Returns:
            List of provider configuration dictionaries
        """
        return db.get_providers()
    
    def add_provider(self, config: Dict[str, Any]) -> BaseProvider:
        """
        Add a new provider.
        
        Args:
            config: Provider configuration dictionary
            
        Returns:
            Created BaseProvider instance
        """
        # Add to database
        db.add_provider(
            name=config["name"],
            api_type=config["api_type"],
            endpoint_url=config["endpoint_url"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            is_active=config.get("is_active", 0),
            rate_limit_tps=config.get("rate_limit_tps"),
            max_tokens=config.get("max_tokens")
        )
        
        self._invalidate_cache()
        
        # Get the newly created provider
        providers = db.get_providers()
        latest = providers[-1] if providers else None
        
        if latest:
            return self._factory.create_provider(latest)
        
        raise ValueError("Failed to retrieve newly created provider")
    
    def update_provider(self, provider_id: int, config: Dict[str, Any]) -> BaseProvider:
        """
        Update an existing provider.
        
        Args:
            provider_id: ID of the provider to update
            config: Updated configuration
            
        Returns:
            Updated BaseProvider instance
        """
        db.update_provider(
            provider_id=provider_id,
            name=config["name"],
            api_type=config["api_type"],
            endpoint_url=config["endpoint_url"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            is_active=config.get("is_active", 0),
            rate_limit_tps=config.get("rate_limit_tps", None),
            max_tokens=config.get("max_tokens", None)
        )
        
        self._invalidate_cache()
        
        # Return updated provider
        return self.get_provider_by_id(provider_id)
    
    def delete_provider(self, provider_id: int) -> None:
        """
        Delete a provider.
        
        Args:
            provider_id: ID of the provider to delete
        """
        db.delete_provider(provider_id)
        self._invalidate_cache()
    
    def set_active_provider(self, provider_id: int) -> None:
        """
        Set a provider as the active one.
        
        Args:
            provider_id: ID of the provider to activate
        """
        db.set_active_provider(provider_id)
        self._invalidate_cache()
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported provider types."""
        return self._factory.get_supported_types()
    
    def reload_active_provider(self) -> Optional[BaseProvider]:
        """Force reload of the active provider."""
        self._invalidate_cache()
        return self.get_active_provider()


# Global service instance
_provider_service: Optional[ProviderService] = None


def get_provider_service() -> ProviderService:
    """Get the global provider service instance."""
    global _provider_service
    if _provider_service is None:
        _provider_service = ProviderService()
    return _provider_service