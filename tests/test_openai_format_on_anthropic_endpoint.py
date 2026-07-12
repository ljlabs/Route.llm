"""
Tests for OpenAI-format content blocks on the Anthropic endpoint.

Verifies that when an AI agent sends OpenAI-format requests (with image_url blocks)
to the /v1/messages Anthropic endpoint, the image and PDF content is correctly
preserved through the translation pipeline.
"""

import json
import pytest
from core.providers.translation import (
    anthropic_to_openai_request,
    openai_to_anthropic_request,
)
from core.providers.gemini import GeminiProvider


# Minimal valid base64-encoded 1x1 white pixel PNG
TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

# Minimal valid base64 for a PDF header
TINY_PDF = "JVBERi0xLjQK"


@pytest.fixture
def gemini_provider():
    return GeminiProvider(
        name="Google AI Studio (Gemma 4)",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key="test-api-key",
        model_name="gemma-4-31b-it",
    )


class TestImageUrlPassthroughInAnthropicToOpenai:
    """Verify that image_url blocks survive anthropic_to_openai_request translation."""

    def test_image_url_block_preserved(self):
        """An image_url block in Anthropic content should pass through to OpenAI format."""
        anth_req = {
            "model": "gemma-4-31b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TINY_PNG}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }

        openai_req = anthropic_to_openai_request(anth_req, "gemma-4-31b-it")

        msg = openai_req["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2

        # Text block preserved
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Describe this image"

        # image_url block preserved as-is
        img = msg["content"][1]
        assert img["type"] == "image_url"
        assert img["image_url"]["url"] == f"data:image/png;base64,{TINY_PNG}"

    def test_image_url_only_no_text(self):
        """A message with only an image_url block should be preserved."""
        anth_req = {
            "model": "gemma-4-31b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{TINY_PNG}"
                            },
                        }
                    ],
                }
            ],
            "max_tokens": 512,
        }

        openai_req = anthropic_to_openai_request(anth_req, "gemma-4-31b-it")

        msg = openai_req["messages"][0]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "image_url"
        assert "data:image/jpeg;base64," in msg["content"][0]["image_url"]["url"]

    def test_multiple_image_url_blocks(self):
        """Multiple image_url blocks should all be preserved."""
        anth_req = {
            "model": "gemma-4-31b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{TINY_PNG}"}},
                    ],
                }
            ],
            "max_tokens": 1024,
        }

        openai_req = anthropic_to_openai_request(anth_req, "gemma-4-31b-it")

        content = openai_req["messages"][0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]
        assert "data:image/webp;base64," in content[2]["image_url"]["url"]

    def test_mixed_anthropic_image_and_openai_image_url(self):
        """Both Anthropic image blocks and OpenAI image_url blocks should be preserved."""
        anth_req = {
            "model": "gemma-4-31b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": TINY_PNG,
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{TINY_PNG}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }

        openai_req = anthropic_to_openai_request(anth_req, "gemma-4-31b-it")

        content = openai_req["messages"][0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        # Anthropic image -> OpenAI image_url (converted)
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]
        # OpenAI image_url -> preserved as-is
        assert content[2]["type"] == "image_url"
        assert "data:image/jpeg;base64," in content[2]["image_url"]["url"]

    def test_system_prompt_with_image_url(self):
        """System prompt + user message with image_url should both work."""
        anth_req = {
            "model": "gemma-4-31b-it",
            "system": "You are a helpful assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TINY_PNG}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }

        openai_req = anthropic_to_openai_request(anth_req, "gemma-4-31b-it")

        assert openai_req["messages"][0]["role"] == "system"
        assert openai_req["messages"][1]["role"] == "user"
        content = openai_req["messages"][1]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[1]["type"] == "image_url"


class TestDetectPdfWithImageUrlFormat:
    """Verify that detect_pdf_content recognizes image_url blocks with PDF data URLs."""

    def test_detects_pdf_image_url(self, gemini_provider):
        """A PDF sent as image_url should be detected as PDF content."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse this PDF"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{TINY_PDF}"
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is True

    def test_does_not_detect_image_png_image_url(self, gemini_provider):
        """A PNG sent as image_url should NOT be detected as PDF."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TINY_PNG}"
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is False

    def test_detects_pdf_in_anthropic_format(self, gemini_provider):
        """A PDF in Anthropic image format should still be detected."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse this PDF"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": TINY_PDF,
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is True


class TestBuildNativePdfRequestWithImageUrl:
    """Verify that build_native_pdf_request handles image_url format PDFs."""

    def test_image_url_pdf_converted_to_inline_data(self, gemini_provider):
        """PDF in image_url format should be converted to Gemini inline_data."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse this recipe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{TINY_PDF}"
                            },
                        },
                    ],
                }
            ]
        }

        native = gemini_provider.build_native_pdf_request(request)

        parts = native["contents"][0]["parts"]
        text_parts = [p for p in parts if "text" in p]
        inline_parts = [p for p in parts if "inline_data" in p]

        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "Parse this recipe"
        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "application/pdf"
        assert inline_parts[0]["inline_data"]["data"] == TINY_PDF

    def test_image_url_image_not_confused_with_pdf(self, gemini_provider):
        """A regular image in image_url format should NOT be treated as PDF inline_data."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TINY_PNG}"
                            },
                        },
                    ],
                }
            ]
        }

        native = gemini_provider.build_native_pdf_request(request)

        parts = native["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inline_data" in p]

        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
        assert inline_parts[0]["inline_data"]["data"] == TINY_PNG

    def test_full_openai_format_request_roundtrip(self, gemini_provider):
        """
        Simulate the actual proxy flow for an OpenAI-format client request:
        1. Client sends image_url format to /v1/messages (Anthropic endpoint)
        2. Router calls detect_pdf_content on the ORIGINAL request
        3. Router calls build_native_pdf_request on the ORIGINAL request
        4. The native request is sent to Gemini's generateContent endpoint
        """
        openai_format_request = {
            "model": "gemma-4-31b-it",
            "system": "You are a recipe parser.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse this recipe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{TINY_PDF}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 8192,
            "temperature": 0.3,
            "stream": False,
        }

        # Step 1: Router detects PDF in the original request
        assert gemini_provider.detect_pdf_content(openai_format_request) is True

        # Step 2: Router builds native request from the original (no translation step)
        native = gemini_provider.build_native_pdf_request(openai_format_request)

        # Verify structure
        assert "contents" in native
        assert len(native["contents"]) == 1
        assert native["contents"][0]["role"] == "user"

        parts = native["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inline_data" in p]
        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "application/pdf"
        assert inline_parts[0]["inline_data"]["data"] == TINY_PDF

        # Verify system instruction
        assert "system_instruction" in native
        assert native["system_instruction"]["parts"][0]["text"] == "You are a recipe parser."

        # Verify generation config
        assert native["generationConfig"]["max_output_tokens"] == 8192
        assert native["generationConfig"]["temperature"] == 0.3
