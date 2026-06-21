"""
Request Models

Pydantic models for API request validation.
"""

from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Union[str, List[Any]] = Field(..., description="Message content")


class Tool(BaseModel):
    """Tool/function definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    input_schema: dict = Field(default={}, description="JSON schema for tool input")


class AnthropicRequest(BaseModel):
    """Anthropic /v1/messages API request model."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="Chat messages")
    system: Optional[Union[str, List[Any]]] = Field(default=None, description="System prompt")
    tools: Optional[List[Tool]] = Field(default=None, description="Available tools")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming")


class OpenAIRequest(BaseModel):
    """OpenAI /v1/chat/completions API request model."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="Chat messages")
    tools: Optional[List[Tool]] = Field(default=None, description="Available tools")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming")


class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings API request model."""
    model: Optional[str] = Field(default=None, description="Model identifier (optional if active embedding provider is set)")
    input: Union[str, List[str]] = Field(..., description="Text(s) to generate embeddings for")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format (float or base64)")


class ChatTestRequest(BaseModel):
    """Model for /api/chat test endpoint."""
    message: str = Field(..., description="User message to send")


class SettingsRequest(BaseModel):
    """Model for settings updates."""
    log_limit: Optional[int] = Field(default=None, description="Maximum log entries to keep")
    rate_limit_tps: Optional[float] = Field(default=None, description="Rate limit in requests per second")
    max_tokens: Optional[int] = Field(default=None, description="Default max tokens for all providers")