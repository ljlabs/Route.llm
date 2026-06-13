"""
Custom Exception Hierarchy

All custom exceptions for the proxy system.
"""


class ProxyException(Exception):
    """Base exception for all proxy-related errors."""
    pass


class ConfigurationException(ProxyException):
    """Errors related to provider configuration."""
    pass


class ProviderNotFoundException(ConfigurationException):
    """Raised when a requested provider is not found."""
    pass


class InvalidConfigurationException(ConfigurationException):
    """Raised when provider configuration is invalid."""
    pass


class TranslationException(ProxyException):
    """Base exception for translation errors."""
    pass


class RequestTranslationException(TranslationException):
    """Raised when request translation fails."""
    pass


class ResponseTranslationException(TranslationException):
    """Raised when response translation fails."""
    pass


class ConnectionException(ProxyException):
    """Base exception for connection errors."""
    pass


class BackendUnavailableException(ConnectionException):
    """Raised when backend API is unavailable."""
    pass


class RateLimitExceededException(ConnectionException):
    """Raised when rate limit is exceeded."""
    pass


class ValidationException(ProxyException):
    """Raised when request validation fails."""
    pass