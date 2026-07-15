import pytest
from core.providers.translation import (
    _openai_image_url_to_anthropic,
    _anthropic_image_to_openai,
    openai_to_anthropic_request,
    anthropic_to_openai_response,
    openai_to_anthropic_response
)
from models.response import ContentBlock


class TestOpenAIImageURLToAnthropic:
    """Tests for converting OpenAI image_url blocks to Anthropic format."""
    
    def test_http_url_image(self):
        """HTTP URL images should use source.type='url', not 'base64'."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/image.jpg"
        assert "media_type" not in result["source"]
    
    def test_data_uri_image(self):
        """Data URI images should use source.type='base64'."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc123"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"
    
    def test_data_uri_with_semicolon(self):
        """Data URI with additional parameters."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg; charset=utf-8;base64,def456"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "def456"
    
    def test_https_url(self):
        """HTTPS URLs should also be handled."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"].startswith("https://")
    
    def test_http_url(self):
        """HTTP URLs should also be handled."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"].startswith("http://")
    
    def test_fallback_for_unknown_format(self):
        """Unknown format should fall back to base64."""
        openai_part = {
            "type": "image_url",
            "image_url": {"url": "some-other-format"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["data"] == "some-other-format"
    
    def test_input_image_type(self):
        """input_image type (Responses API) should be handled the same way."""
        openai_part = {
            "type": "input_image",
            "image_url": {"url": "https://example.com/image.png"}
        }
        result = _openai_image_url_to_anthropic(openai_part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"


class TestAnthropicImageToOpenAI:
    """Tests for converting Anthropic image blocks to OpenAI format."""
    
    def test_base64_source(self):
        """Base64 source should be converted to data URI."""
        anthropic_part = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123"
            }
        }
        result = _anthropic_image_to_openai(anthropic_part)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "data:image/png;base64,abc123"
    
    def test_url_source(self):
        """URL source should be passed through."""
        anthropic_part = {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.jpg"
            }
        }
        result = _anthropic_image_to_openai(anthropic_part)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/image.jpg"
    
    def test_missing_source_type_defaults_to_base64(self):
        """Missing source type should default to base64."""
        anthropic_part = {
            "type": "image",
            "source": {
                "media_type": "image/jpeg",
                "data": "def456"
            }
        }
        result = _anthropic_image_to_openai(anthropic_part)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "data:image/jpeg;base64,def456"


class TestOpenAIToAnthropicRequest:
    """Tests for OpenAI → Anthropic request translation with images."""
    
    def test_mixed_content_with_urls(self):
        """Mixed content (text + URL image + data URI) should preserve all parts."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        }
        result = openai_to_anthropic_request(openai_request, "claude-3-5-sonnet")
        content = result["messages"][0]["content"]
        
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "url"
        assert content[2]["type"] == "image"
        assert content[2]["source"]["type"] == "base64"
    
    def test_tool_result_with_image_url(self):
        """Tool results with image URLs should be preserved."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": [
                        {"type": "text", "text": "Result"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/result.jpg"}}
                    ]
                }
            ]
        }
        result = openai_to_anthropic_request(openai_request, "claude-3-5-sonnet")
        # Find the tool_result message
        for msg in result["messages"]:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "tool_result":
                        tool_content = part["content"]
                        # Should have text and image
                        assert len(tool_content) == 2
                        assert tool_content[0]["type"] == "text"
                        assert tool_content[1]["type"] == "image"
                        assert tool_content[1]["source"]["type"] == "url"
                        return
        pytest.fail("Tool result with image not found")


class TestAnthropicToOpenAIResponse:
    """Tests for Anthropic → OpenAI response translation with images."""
    
    def test_anthropic_response_with_text_only(self):
        """Text-only response should have string content."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "text", "text": "Hello"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        result = anthropic_to_openai_response(anthropic_response)
        message = result["choices"][0]["message"]
        
        assert isinstance(message["content"], str)
        assert message["content"] == "Hello"
    
    def test_anthropic_response_with_image(self):
        """Images in Anthropic responses should be preserved in OpenAI format."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "text", "text": "Here's the image:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc123"
                    }
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        result = anthropic_to_openai_response(anthropic_response)
        message = result["choices"][0]["message"]
        
        # Content should be a list, not a string
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in message["content"][1]["image_url"]["url"]
    
    def test_anthropic_response_with_url_image(self):
        """URL images should be preserved as-is."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "text", "text": "Image:"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.jpg"
                    }
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        result = anthropic_to_openai_response(anthropic_response)
        message = result["choices"][0]["message"]
        
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2
        assert message["content"][1]["type"] == "image_url"
        assert message["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
    
    def test_empty_content(self):
        """Empty content should result in null content."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        result = anthropic_to_openai_response(anthropic_response)
        message = result["choices"][0]["message"]
        assert message["content"] is None


class TestOpenAIToAnthropicResponse:
    """Tests for OpenAI → Anthropic response translation with images."""
    
    def test_openai_response_with_string_content(self):
        """String content should be converted to text block."""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        result = openai_to_anthropic_response(openai_response)
        content = result["content"]
        
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"
    
    def test_openai_response_with_image_list(self):
        """List content with images should be preserved."""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here's an image:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        result = openai_to_anthropic_response(openai_response)
        content = result["content"]
        
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Here's an image:"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "url"
        assert content[1]["source"]["url"] == "https://example.com/image.jpg"
    
    def test_openai_response_with_data_uri_image(self):
        """Data URI images should be converted correctly."""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Image:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        result = openai_to_anthropic_response(openai_response)
        content = result["content"]
        
        assert len(content) == 2
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"
        assert content[1]["source"]["data"] == "abc123"


class TestContentBlockModel:
    """Tests for ContentBlock model validation with images."""
    
    def test_text_block(self):
        """Text block should be valid."""
        block = ContentBlock(type="text", text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"
    
    def test_tool_use_block(self):
        """Tool use block should be valid."""
        block = ContentBlock(type="tool_use", id="tool_123", name="test", input={"arg": "val"})
        assert block.type == "tool_use"
        assert block.id == "tool_123"
        assert block.name == "test"
        assert block.input == {"arg": "val"}
    
    def test_image_block_with_source(self):
        """Image block with source should be valid."""
        block = ContentBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123"
            }
        )
        assert block.type == "image"
        assert block.source["type"] == "base64"
        assert block.source["media_type"] == "image/png"
        assert block.source["data"] == "abc123"
    
    def test_image_block_with_url_source(self):
        """Image block with URL source should be valid."""
        block = ContentBlock(
            type="image",
            source={
                "type": "url",
                "url": "https://example.com/image.jpg"
            }
        )
        assert block.type == "image"
        assert block.source["type"] == "url"
        assert block.source["url"] == "https://example.com/image.jpg"
    
    def test_optional_fields(self):
        """Optional fields should be None by default."""
        block = ContentBlock(type="text")
        assert block.text is None
        assert block.id is None
        assert block.name is None
        assert block.input is None
        assert block.source is None