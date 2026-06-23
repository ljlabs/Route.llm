"""
Response format schemas for validating SSE events against Anthropic and OpenAI specifications.

Used for runtime validation and logging when response_format is configured.
"""

from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Anthropic SSE event schemas (per https://docs.anthropic.com/en/api/messages-streaming)
# ---------------------------------------------------------------------------

class AnthropicMessageStartData(BaseModel):
    type: Literal["message_start"]
    message: Dict[str, Any]


class AnthropicContentBlockStartData(BaseModel):
    type: Literal["content_block_start"]
    index: int
    content_block: Dict[str, Any]


class AnthropicContentBlockDeltaData(BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: Dict[str, Any]


class AnthropicContentBlockStopData(BaseModel):
    type: Literal["content_block_stop"]
    index: int


class AnthropicMessageDeltaData(BaseModel):
    type: Literal["message_delta"]
    delta: Dict[str, Any]


class AnthropicMessageStopData(BaseModel):
    type: Literal["message_stop"]


class AnthropicPingData(BaseModel):
    type: Literal["ping"]


class AnthropicErrorData(BaseModel):
    type: Literal["error"]
    error: Dict[str, Any]


ANTHROPIC_EVENT_VALIDATORS = {
    "message_start": AnthropicMessageStartData,
    "content_block_start": AnthropicContentBlockStartData,
    "content_block_delta": AnthropicContentBlockDeltaData,
    "content_block_stop": AnthropicContentBlockStopData,
    "message_delta": AnthropicMessageDeltaData,
    "message_stop": AnthropicMessageStopData,
    "ping": AnthropicPingData,
    "error": AnthropicErrorData,
}

ANTHROPIC_VALID_EVENT_TYPES = set(ANTHROPIC_EVENT_VALIDATORS.keys())


# ---------------------------------------------------------------------------
# OpenAI SSE event schemas (per OpenAI Chat Completions streaming spec)
# ---------------------------------------------------------------------------

class OpenAIChoiceDelta(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[OpenAIChoiceDelta]


OPENAI_VALID_OBJECT_TYPES = {"chat.completion.chunk"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_anthropic_sse_event(event_type: str, data: Dict[str, Any]) -> bool:
    """Validate an Anthropic SSE event against known schemas.

    Returns True if valid, False if the event doesn't match any known type.
    Logs a warning via the logging module on failure.
    """
    import logging
    logger = logging.getLogger(__name__)

    if event_type not in ANTHROPIC_EVENT_VALIDATORS:
        logger.warning(
            "SSE_VALIDATION: Unknown Anthropic event type '%s'. "
            "Known types: %s",
            event_type, sorted(ANTHROPIC_VALID_EVENT_TYPES),
        )
        return False

    validator = ANTHROPIC_EVENT_VALIDATORS[event_type]
    try:
        validator.model_validate(data)
        return True
    except Exception as e:
        logger.warning(
            "SSE_VALIDATION: Anthropic event type '%s' failed schema validation: %s",
            event_type, e,
        )
        return False


def validate_anthropic_sse_line(line: str) -> bool:
    """Validate a full Anthropic SSE line (the data: portion after extracting event type).

    Parses the 'event:' and 'data:' lines and validates the data payload.
    Returns True if valid.
    """
    import json
    import logging
    logger = logging.getLogger(__name__)

    if not line.startswith("data:"):
        return True

    data_content = line[len("data:"):].strip()
    if not data_content or data_content == "[DONE]":
        return True

    try:
        data = json.loads(data_content)
    except json.JSONDecodeError as e:
        logger.warning("SSE_VALIDATION: Failed to parse Anthropic data JSON: %s", e)
        return False

    event_type = data.get("type", "")
    return validate_anthropic_sse_event(event_type, data)


def validate_openai_sse_line(line: str) -> bool:
    """Validate an OpenAI SSE data line against the chat.completion.chunk schema.

    Returns True if valid.
    """
    import json
    import logging
    logger = logging.getLogger(__name__)

    if not line.startswith("data:"):
        return True

    data_content = line[len("data:"):].strip()
    if not data_content or data_content == "[DONE]":
        return True

    try:
        data = json.loads(data_content)
    except json.JSONDecodeError as e:
        logger.warning("SSE_VALIDATION: Failed to parse OpenAI data JSON: %s", e)
        return False

    obj_type = data.get("object", "")
    if obj_type not in OPENAI_VALID_OBJECT_TYPES:
        logger.warning(
            "SSE_VALIDATION: Unknown OpenAI object type '%s'. Expected one of: %s",
            obj_type, sorted(OPENAI_VALID_OBJECT_TYPES),
        )
        return False

    try:
        OpenAIChatCompletionChunk.model_validate(data)
        return True
    except Exception as e:
        logger.warning(
            "SSE_VALIDATION: OpenAI chunk failed schema validation: %s", e,
        )
        return False
