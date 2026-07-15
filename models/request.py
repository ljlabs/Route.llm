"""
Request Models

Pydantic models for API request validation.
"""

from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    """A message accepted by either OpenAI or Anthropic compatibility routes."""
    role: str = Field(..., description="Message role: system, user, assistant, tool")
    content: Optional[Union[str, List[Any]]] = Field(default=None, description="Message content")
    tool_calls: Optional[List[Any]] = Field(default=None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID for tool response messages")
    name: Optional[str] = Field(default=None, description="Name for function calling (legacy)")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in {"system", "user", "assistant", "tool", "developer"}:
            raise ValueError("role must be one of system, user, assistant, tool, or developer")
        return value


class FlexibleTool(BaseModel):
    """Tool definition that accepts both OpenAI and Anthropic formats."""
    # OpenAI format fields
    type: Optional[str] = Field(default="function", description="Tool type")
    function: Optional[dict] = Field(default=None, description="Function definition (OpenAI format)")
    # Anthropic format fields
    name: Optional[str] = Field(default=None, description="Tool name")
    description: Optional[str] = Field(default="", description="Tool description")
    input_schema: Optional[dict] = Field(default=None, description="JSON schema for tool input")

    @model_validator(mode='after')
    def normalize_openai_format(self):
        """Normalize OpenAI format to Anthropic-compatible format."""
        if self.function and not self.name:
            self.name = self.function.get('name')
            if not self.description or self.description == "":
                self.description = self.function.get('description', "")
            if not self.input_schema:
                self.input_schema = self.function.get('parameters', {})
        return self


class AnthropicRequest(BaseModel):
    """Anthropic /v1/messages API request model."""
    model: str = Field(..., min_length=1, description="Model identifier")
    messages: List[Message] = Field(..., min_length=1, description="Chat messages")
    system: Optional[Union[str, List[Any]]] = Field(default=None, description="System prompt")
    tools: Optional[List[FlexibleTool]] = Field(default=None, description="Available tools")
    max_tokens: int = Field(..., gt=0, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming")
    # Additional Anthropic parameters for full compatibility
    top_k: Optional[int] = Field(default=None, ge=1, le=500, description="Top-k sampling")
    meta: Optional[dict] = Field(default=None, description="User metadata")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Custom stop sequences")
    tool_choice: Optional[dict] = Field(default=None, description="Tool selection control")
    thinking: Optional[dict] = Field(default=None, description="Extended thinking config")
    metadata: Optional[dict] = Field(default=None, description="Request metadata")

    @model_validator(mode="after")
    def validate_message_sequence(self):
        """Normalize Claude-style inline system messages before routing."""
        inline_system = [message.content for message in self.messages if message.role == "system"]
        self.messages = [message for message in self.messages if message.role != "system"]

        if inline_system:
            system_segments = [self.system, *inline_system]
            if any(isinstance(segment, list) for segment in system_segments if segment is not None):
                system_blocks = []
                for segment in system_segments:
                    if isinstance(segment, list):
                        system_blocks.extend(segment)
                    elif segment is not None:
                        system_blocks.append({"type": "text", "text": str(segment)})
                self.system = system_blocks
            else:
                self.system = "\n".join(str(segment) for segment in system_segments if segment is not None)

        if not self.messages:
            raise ValueError("Anthropic requests must include at least one user message")
        if self.messages[0].role != "user":
            raise ValueError("Anthropic conversations must begin with a user message")
        if any(message.role not in {"user", "assistant"} for message in self.messages):
            raise ValueError("Anthropic messages only support user and assistant roles after system normalization")
        return self


class OpenAIRequest(BaseModel):
    """OpenAI /v1/chat/completions API request model."""
    model: str = Field(..., min_length=1, description="Model identifier")
    messages: List[Message] = Field(..., min_length=1, description="Chat messages")
    tools: Optional[List[FlexibleTool]] = Field(default=None, description="Available tools")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming")
    # Additional OpenAI parameters for full compatibility
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="Frequency penalty")
    logit_bias: Optional[dict] = Field(default=None, description="Token bias map")
    user: Optional[str] = Field(default=None, description="End-user identifier")
    seed: Optional[int] = Field(default=None, description="System fingerprint seed")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Tool selection control")
    response_format: Optional[dict] = Field(default=None, description="Response format (JSON mode)")
    stream_options: Optional[dict] = Field(default=None, description="Streaming response options")
    functions: Optional[List[dict]] = Field(default=None, description="Legacy function calling")


class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings API request model."""
    model: Optional[str] = Field(default=None, description="Model identifier (optional if active embedding provider is set)")
    input: Union[str, List[str]] = Field(..., description="Text(s) to generate embeddings for")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format (float or base64)")
    input_type: Optional[str] = Field(default=None, description="Input type for providers that require it (e.g. 'query' or 'passage' for Nvidia NIM)")
    truncate: Optional[str] = Field(default=None, description="Truncation strategy (e.g. 'NONE', 'START', 'END' for Nvidia NIM)")


class ChatTestRequest(BaseModel):
    """Model for /api/chat test endpoint."""
    message: str = Field(..., description="User message to send")


class SettingsRequest(BaseModel):
    """Model for settings updates."""
    log_limit: Optional[int] = Field(default=None, description="Maximum log entries to keep")
    rate_limit_tps: Optional[float] = Field(default=None, description="Rate limit in requests per second")
    max_tokens: Optional[int] = Field(default=None, description="Default max tokens for all providers")
    response_format: Optional[str] = Field(default=None, description="Response format: 'anthropic' or 'openai'")
    disable_streaming: Optional[bool] = Field(default=None, description="Disable streaming responses (override stream flag to false)")