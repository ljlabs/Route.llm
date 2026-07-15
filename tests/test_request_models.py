"""
Tests for request model validation.

Ensures backward compatibility with Anthropic-converted tool format
and support for native OpenAI tool format.
"""

import pytest
from models.request import OpenAIRequest, AnthropicRequest, FlexibleTool


def test_flexible_tool_anthropic_format():
    """Test FlexibleTool accepts Anthropic-converted format (existing backward compat)."""
    tool_data = {
        "name": "bash",
        "description": "Execute a bash command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    }
    
    tool = FlexibleTool(**tool_data)
    
    # Normalized fields should be populated
    assert tool.name == "bash"
    assert tool.description == "Execute a bash command"
    assert tool.input_schema["type"] == "object"
    assert tool.function is None  # OpenAI format field should be None
    assert tool.type == "function"  # Default


def test_flexible_tool_openai_format():
    """Test FlexibleTool accepts native OpenAI format (new mini-swe use case)."""
    tool_data = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
    
    tool = FlexibleTool(**tool_data)
    
    # Normalized fields should be extracted from function
    assert tool.name == "bash"
    assert tool.description == "Execute a bash command"
    assert tool.input_schema["type"] == "object"
    assert tool.type == "function"
    assert tool.function is not None


def test_openai_request_with_anthropic_tools():
    """Test OpenAI request accepts Anthropic-converted tools (backward compat)."""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Execute ls"}
        ],
        "tools": [
            {
                "name": "bash",
                "description": "Execute a bash command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
    
    request = OpenAIRequest(**request_data)
    
    assert request.model == "gpt-4o"
    assert len(request.tools) == 1
    assert request.tools[0].name == "bash"
    assert request.tools[0].input_schema["type"] == "object"


def test_openai_request_with_native_openai_tools():
    """Test OpenAI request accepts native OpenAI tools (mini-swe use case)."""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Execute ls"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    }
    
    request = OpenAIRequest(**request_data)
    
    assert request.model == "gpt-4o"
    assert len(request.tools) == 1
    assert request.tools[0].name == "bash"
    assert request.tools[0].input_schema["type"] == "object"
    assert request.tools[0].function is not None


def test_openai_request_mixed_tools():
    """Test OpenAI request can handle mixed tool formats (edge case)."""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Execute commands"}
        ],
        "tools": [
            # Anthropic format
            {
                "name": "bash",
                "description": "Execute bash",
                "input_schema": {"type": "object"}
            },
            # Native OpenAI format
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Execute Python",
                    "parameters": {"type": "object"}
                }
            }
        ]
    }
    
    request = OpenAIRequest(**request_data)
    
    assert len(request.tools) == 2
    assert request.tools[0].name == "bash"
    assert request.tools[1].name == "python"


def test_openai_request_without_tools():
    """Test OpenAI request works without tools."""
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    request = OpenAIRequest(**request_data)
    
    assert request.model == "gpt-4o"
    assert request.tools is None


def test_anthropic_request_with_tools():
    """Test Anthropic request still validates correctly."""
    request_data = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "Execute ls"}
        ],
        "tools": [
            {
                "name": "bash",
                "description": "Execute a bash command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        ]
    }
    
    request = AnthropicRequest(**request_data)
    
    assert request.model == "claude-3-5-sonnet"
    assert len(request.tools) == 1
    assert request.tools[0].name == "bash"
    assert request.tools[0].input_schema["type"] == "object"


def test_anthropic_inline_system_messages_are_normalized():
    """Claude-style inline system messages become an ordered top-level prompt."""
    request = AnthropicRequest(
        model="claude-sonnet-4-6",
        max_tokens=32,
        system="Top-level instruction",
        messages=[
            {"role": "system", "content": "First inline instruction"},
            {"role": "system", "content": [{"type": "text", "text": "Second inline instruction"}]},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello"},
        ],
    )

    assert request.system == [
        {"type": "text", "text": "Top-level instruction"},
        {"type": "text", "text": "First inline instruction"},
        {"type": "text", "text": "Second inline instruction"},
    ]
    assert [message.role for message in request.messages] == ["user", "assistant"]
