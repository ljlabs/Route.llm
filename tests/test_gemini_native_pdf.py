"""
Tests for Gemini native PDF support.

Validates that PDF requests are correctly converted from OpenAI/Anthropic format
to Gemini's native generateContent format with inline_data.
"""

import json
import pytest
from core.providers.gemini import GeminiProvider
from core.providers.translation import openai_to_anthropic_request


@pytest.fixture
def gemini_provider():
    return GeminiProvider(
        name="Google AI Studio (Gemma 4 31b)",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key="test-api-key",
        model_name="gemma-4-31b-it",
    )


class TestNativePdfDetection:
    """Test PDF content detection in Anthropic-format requests."""

    def test_detects_application_pdf(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is True

    def test_does_not_detect_image_png(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is False

    def test_does_not_detect_text_only(self, gemini_provider):
        request = {
            "messages": [{"role": "user", "content": "Hello world"}]
        }
        assert gemini_provider.detect_pdf_content(request) is False

    def test_detects_pdf_in_tool_result(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": "JVBERi0xLjQK",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is True

    def test_detects_pdf_among_multiple_images(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "/9j/4AAQ",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                    ],
                }
            ]
        }
        assert gemini_provider.detect_pdf_content(request) is True


class TestNativePdfEndpoint:
    """Test native PDF endpoint derivation."""

    def test_builds_correct_endpoint(self, gemini_provider):
        endpoint = gemini_provider.build_native_pdf_endpoint()
        assert (
            endpoint
            == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent"
        )

    def test_endpoint_from_different_model(self):
        provider = GeminiProvider(
            name="test",
            endpoint_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            api_key="key",
            model_name="gemini-2.5-flash",
        )
        assert (
            provider.build_native_pdf_endpoint()
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        )


class TestNativePdfRequestFormat:
    """Test that native PDF requests match the format Google's generateContent API expects."""

    def test_full_client_request_roundtrip(self, gemini_provider):
        """
        Simulate the full flow: OpenAI client request -> router translates to
        Anthropic -> GeminiProvider builds native request. Assert the output
        matches the format Google's generateContent API expects.
        """
        # The actual OpenAI request from the client
        openai_request = {
            "model": "local",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a recipe parser. Extract the following from the provided image/PDF of a recipe:\n- title: The recipe name\n- description: A brief description\n- servings: Number of servings (integer)\n- ingredients: Array of { name, quantity (number), unit (string) }\n- instructions: Array of { step (1-indexed), text }\n\nReturn ONLY valid JSON, no markdown fences.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Parse this recipe image/PDF and extract the recipe data as JSON.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:application/pdf;base64,JVBERi0xLjUNJeLjz9MKGjI="
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 8192,
            "temperature": 0.3,
            "stream": False,
        }

        # Step 1: Router translates OpenAI -> Anthropic (intermediate representation)
        anthropic_request = openai_to_anthropic_request(
            openai_request, gemini_provider.model_name
        )

        # Step 2: GeminiProvider builds native PDF request
        native_request = gemini_provider.build_native_pdf_request(anthropic_request)

        # Assert the output matches Google's generateContent format
        # Reference: curl from the user's research
        expected = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Parse this recipe image/PDF and extract the recipe data as JSON."
                        },
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": "JVBERi0xLjUNJeLjz9MKGjI=",
                            }
                        },
                    ],
                }
            ],
            "system_instruction": {
                "parts": [
                    {
                        "text": "You are a recipe parser. Extract the following from the provided image/PDF of a recipe:\n- title: The recipe name\n- description: A brief description\n- servings: Number of servings (integer)\n- ingredients: Array of { name, quantity (number), unit (string) }\n- instructions: Array of { step (1-indexed), text }\n\nReturn ONLY valid JSON, no markdown fences."
                    }
                ]
            },
            "generationConfig": {"max_output_tokens": 8192, "temperature": 0.3},
        }

        assert native_request == expected

    def test_contents_structure_matches_generate_content(self, gemini_provider):
        """Assert contents is an array of {role, parts} objects per the Gemini API."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this PDF"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "dGVzdA==",
                            },
                        },
                    ],
                }
            ]
        }
        native = gemini_provider.build_native_pdf_request(request)

        assert isinstance(native["contents"], list)
        assert len(native["contents"]) == 1
        assert native["contents"][0]["role"] == "user"
        assert isinstance(native["contents"][0]["parts"], list)

    def test_inline_data_format_matches_generate_content(self, gemini_provider):
        """Assert inline_data has the correct mime_type and data fields."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Parse"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "abc123",
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

        inline = inline_parts[0]["inline_data"]
        assert "mime_type" in inline
        assert "data" in inline
        assert inline["mime_type"] == "application/pdf"
        assert inline["data"] == "abc123"

    def test_system_prompt_becomes_system_instruction(self, gemini_provider):
        """Assert system prompt is converted to system_instruction with parts."""
        request = {
            "system": "Be concise.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)

        assert "system_instruction" in native
        assert "parts" in native["system_instruction"]
        assert native["system_instruction"]["parts"][0]["text"] == "Be concise."

    def test_no_system_prompt_omits_system_instruction(self, gemini_provider):
        """Assert system_instruction is absent when no system prompt is provided."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert "system_instruction" not in native

    def test_generation_config_includes_max_tokens(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Go"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert native["generationConfig"]["max_output_tokens"] == 4096

    def test_generation_config_includes_temperature(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Go"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.7,
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert native["generationConfig"]["temperature"] == 0.7

    def test_generation_config_includes_top_p(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Go"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
            "top_p": 0.9,
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert native["generationConfig"]["top_p"] == 0.9

    def test_omits_generation_config_when_no_params(self, gemini_provider):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Go"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert "generationConfig" not in native

    def test_consecutive_same_role_messages_are_merged(self, gemini_provider):
        """Gemini requires alternating user/model roles; consecutive same-role must merge."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "a",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "second"}],
                },
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)

        # Should merge into a single user message with 3 parts
        assert len(native["contents"]) == 1
        assert native["contents"][0]["role"] == "user"
        assert len(native["contents"][0]["parts"]) == 3

    def test_multi_turn_preserves_alternation(self, gemini_provider):
        """Multi-turn conversations should alternate user/model roles."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "follow-up"}],
                },
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)

        assert len(native["contents"]) == 3
        assert native["contents"][0]["role"] == "user"
        assert native["contents"][1]["role"] == "model"
        assert native["contents"][2]["role"] == "user"

    def test_assistant_role_maps_to_model(self, gemini_provider):
        """OpenAI 'assistant' role should map to Gemini 'model' role."""
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "response"}],
                },
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)
        assert native["contents"][0]["role"] == "model"

    def test_empty_text_parts_still_produce_text_block(self, gemini_provider):
        """A text part with empty string should still produce a text block."""
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "x",
                            },
                        },
                    ],
                }
            ],
        }
        native = gemini_provider.build_native_pdf_request(request)
        parts = native["contents"][0]["parts"]
        text_parts = [p for p in parts if "text" in p]
        assert len(text_parts) == 1


class TestNativePdfResponseTranslation:
    """Test translation of Gemini generateContent response to Anthropic format."""

    def test_translates_text_response(self, gemini_provider):
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"title": "Chocolate Cake", "servings": 8}'}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 42,
            },
        }
        result = gemini_provider.translate_native_response(gemini_response)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "gemma-4-31b-it"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == '{"title": "Chocolate Cake", "servings": 8}'
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 150
        assert result["usage"]["output_tokens"] == 42

    def test_translates_stop_to_end_turn(self, gemini_provider):
        response = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {},
        }
        result = gemini_provider.translate_native_response(response)
        assert result["stop_reason"] == "end_turn"

    def test_translates_max_tokens(self, gemini_provider):
        response = {
            "candidates": [{"content": {"parts": [{"text": "truncated"}]}, "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {},
        }
        result = gemini_provider.translate_native_response(response)
        assert result["stop_reason"] == "max_tokens"

    def test_handles_empty_candidates(self, gemini_provider):
        response = {"candidates": [], "usageMetadata": {}}
        result = gemini_provider.translate_native_response(response)
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"

    def test_handles_missing_usage(self, gemini_provider):
        response = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        }
        result = gemini_provider.translate_native_response(response)
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_generates_message_id(self, gemini_provider):
        response = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {},
        }
        result = gemini_provider.translate_native_response(response)
        assert result["id"].startswith("msg_")

    def test_multi_part_text_response(self, gemini_provider):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Part 1. "},
                            {"text": "Part 2."},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        }
        result = gemini_provider.translate_native_response(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["text"] == "Part 1. "
        assert result["content"][1]["text"] == "Part 2."
