"""
Response Models

Pydantic models for API response validation.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    """Content block in Anthropic response."""
    type: str = Field(..., description="Block type: text, tool_use, image")
    text: Optional[str] = Field(default=None, description="Text content")
    id: Optional[str] = Field(default=None, description="Tool use ID")
    name: Optional[str] = Field(default=None, description="Tool name")
    input: Optional[dict] = Field(default=None, description="Tool input")
    source: Optional[dict] = Field(default=None, description="Source for image blocks (type, media_type, data/url)")


class Usage(BaseModel):
    """Token usage information."""
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens generated")


class AnthropicResponse(BaseModel):
    """Anthropic /v1/messages API response model."""
    id: str = Field(..., description="Message ID")
    type: str = Field(default="message", description="Response type")
    role: str = Field(default="assistant", description="Response role")
    content: List[ContentBlock] = Field(..., description="Content blocks")
    model: str = Field(..., description="Model used")
    stop_reason: Optional[str] = Field(default=None, description="Reason for stopping")
    stop_sequence: Optional[str] = Field(default=None, description="Stop sequence if any")
    usage: Usage = Field(default_factory=Usage, description="Token usage")


class ChoiceDelta(BaseModel):
    """Choice delta for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    refusal: Optional[str] = None


class Choice(BaseModel):
    """Chat completion choice."""
    index: int = Field(default=0, description="Choice index")
    message: Optional[dict] = Field(default=None, description="Complete message")
    delta: Optional[ChoiceDelta] = Field(default=None, description="Streaming delta")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")
    logprobs: Optional[dict] = Field(default=None, description="Log probabilities")


class UsageInfo(BaseModel):
    """OpenAI-style usage information."""
    prompt_tokens: int = Field(default=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, description="Completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")


class OpenAIResponse(BaseModel):
    """OpenAI /v1/chat/completions API response model."""
    id: str = Field(..., description="Completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Choice] = Field(..., description="Completion choices")
    usage: UsageInfo = Field(default_factory=UsageInfo, description="Token usage")
    system_fingerprint: Optional[str] = Field(default=None, description="System fingerprint")


class ChatTestResponse(BaseModel):
    """Response for /api/chat test endpoint."""
    response: str = Field(..., description="Model's response text")
    provider: str = Field(..., description="Provider used")


class ErrorDetail(BaseModel):
    """Error details."""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    param: Optional[str] = Field(default=None, description="Parameter that caused the error")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail


class AnthropicErrorResponse(BaseModel):
    """Anthropic-compatible error response."""
    type: str = Field(default="error", description="Error type")
    error: ErrorDetail