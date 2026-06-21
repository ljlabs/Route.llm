"""
Provider Models

Pydantic models for provider configuration validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Provider configuration model."""
    id: Optional[int] = None
    name: str = Field(..., description="Provider display name")
    api_type: str = Field(..., description="API type: openai, anthropic, gemini, mistral, openrouter")
    endpoint_url: str = Field(..., description="API endpoint URL")
    api_key: str = Field(..., description="API key for authentication")
    model_name: str = Field(..., description="Model identifier")
    is_active: bool = Field(default=False, description="Whether this is the active provider")
    rate_limit_tps: Optional[float] = Field(default=None, description="Provider-specific rate limit in TPS")
    max_tokens: Optional[int] = Field(default=None, description="Provider-specific max tokens override")


class ProviderCreate(BaseModel):
    """Model for creating a new provider."""
    name: str = Field(..., description="Provider display name")
    api_type: str = Field(..., description="API type")
    endpoint_url: str = Field(..., description="API endpoint URL")
    api_key: str = Field(..., description="API key")
    model_name: str = Field(..., description="Model name")
    is_active: bool = Field(default=False, description="Set as active provider")
    rate_limit_tps: Optional[float] = Field(default=None, description="Provider-specific rate limit in TPS")
    max_tokens: Optional[int] = Field(default=None, description="Provider-specific max tokens override")


class ProviderUpdate(BaseModel):
    """Model for updating a provider."""
    name: Optional[str] = None
    api_type: Optional[str] = None
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    is_active: Optional[bool] = None
    rate_limit_tps: Optional[float] = None
    max_tokens: Optional[int] = None


class ProviderResponse(BaseModel):
    """Model for provider API responses."""
    id: int
    name: str
    api_type: str
    endpoint_url: str
    api_key: str
    model_name: str
    is_active: bool
    is_active_embedding: Optional[bool] = False
    rate_limit_tps: Optional[float] = None
    max_tokens: Optional[int] = None
    
    class Config:
        from_attributes = True


class SettingsResponse(BaseModel):
    """Model for settings API responses."""
    log_limit: int
    rate_limit_tps: float
    max_tokens: int


class LogEntry(BaseModel):
    """Model for log entries."""
    id: int
    timestamp: str
    provider_name: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    request_body: Optional[str] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    
    class Config:
        from_attributes = True