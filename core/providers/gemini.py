"""
Gemini Provider Implementation

Extends OpenAI provider with Gemini-specific sanitization and native PDF support.
"""

from typing import Any, Dict, Optional
from .openai import OpenAIProvider
from .translation import sanitize_openai_payload
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiProvider(OpenAIProvider):
    """Provider for Google Gemini API (OpenAI-compatible format)."""

    def __init__(self, **kwargs):
        super().__init__(api_type="gemini", **kwargs)

    @property
    def supports_native_pdf(self) -> bool:
        """Whether this provider supports native PDF via generateContent endpoint."""
        return True

    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to Gemini format with strict sanitization."""
        # Use the parent translation
        wrapped = anthropic_to_openai_request(anthropic_request, self.model_name)
        # Apply Gemini-specific sanitization (allows extra_content for thought signature)
        return sanitize_openai_payload(wrapped, is_gemini=True)

    def sanitize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request specifically for Gemini API."""
        return sanitize_openai_payload(request, is_gemini=True)

    def detect_pdf_content(self, anthropic_request: Dict[str, Any]) -> bool:
        """Detect PDF and generic base64 documents that require Gemini inline_data."""
        def has_native_media(content: Any) -> bool:
            if not isinstance(content, list):
                return False
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "tool_result" and has_native_media(part.get("content")):
                    return True
                if part_type == "image":
                    source = part.get("source") or {}
                    if source.get("media_type") == "application/pdf":
                        return True
                elif part_type == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                    if isinstance(url, str) and url.startswith("data:application/pdf"):
                        return True
                elif part_type == "document":
                    source = part.get("source") or {}
                    if source.get("type") == "base64" and source.get("data"):
                        return True
                elif part_type in {"input_file", "file"}:
                    nested_file = part.get("file") if isinstance(part.get("file"), dict) else {}
                    if (part.get("file_data") or part.get("file_url")
                            or nested_file.get("file_data") or nested_file.get("file_url")):
                        return True
            return False

        return any(has_native_media(message.get("content")) for message in anthropic_request.get("messages", []))

    def build_native_pdf_endpoint(self) -> str:
        """Derive the native generateContent endpoint URL from the OpenAI-compat endpoint."""
        # https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
        # → https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        base = self.endpoint_url.split("/openai/")[0]
        return f"{base}/models/{self.model_name}:generateContent"

    def build_native_pdf_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an Anthropic request containing PDFs to Gemini native generateContent format."""
        contents = []

        # System prompt → system_instruction
        system_instruction = None
        system_content = anthropic_request.get("system")
        if system_content:
            if isinstance(system_content, str):
                system_instruction = {"parts": [{"text": system_content}]}
            elif isinstance(system_content, list):
                text_parts = [p.get("text", "") for p in system_content if isinstance(p, dict) and p.get("type") == "text"]
                if text_parts:
                    system_instruction = {"parts": [{"text": "".join(text_parts)}]}

        for msg in anthropic_request.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                continue

            gemini_role = "user" if role == "user" else "model"
            parts = []

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")

                    if part_type == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part_type == "image":
                        source = part.get("source", {})
                        parts.append({
                            "inline_data": {
                                "mime_type": source.get("media_type", "image/png"),
                                "data": source.get("data", "")
                            }
                        })
                    elif part_type == "document":
                        source = part.get("source", {})
                        if source.get("type") == "base64" and source.get("data"):
                            parts.append({
                                "inline_data": {
                                    "mime_type": source.get("media_type", "application/octet-stream"),
                                    "data": source["data"]
                                }
                            })
                    elif part_type == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, _, data = url.partition(",")
                            mime_type = header.split(":", 1)[1].split(";")[0] if ":" in header else "image/png"
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": data
                                }
                            })
                    elif part_type == "tool_result":
                        tool_content = part.get("content")
                        if isinstance(tool_content, list):
                            for sub in tool_content:
                                if isinstance(sub, dict):
                                    if sub.get("type") == "text":
                                        parts.append({"text": sub.get("text", "")})
                                    elif sub.get("type") == "image":
                                        source = sub.get("source", {})
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": source.get("media_type", "image/png"),
                                                "data": source.get("data", "")
                                            }
                                        })
                                    elif sub.get("type") == "document":
                                        source = sub.get("source", {})
                                        if source.get("type") == "base64" and source.get("data"):
                                            parts.append({
                                                "inline_data": {
                                                    "mime_type": source.get("media_type", "application/octet-stream"),
                                                    "data": source["data"]
                                                }
                                            })
                                    elif sub.get("type") == "image_url":
                                        url = sub.get("image_url", {}).get("url", "")
                                        if url.startswith("data:"):
                                            header, _, data = url.partition(",")
                                            mime_type = header.split(":", 1)[1].split(";")[0] if ":" in header else "image/png"
                                            parts.append({
                                                "inline_data": {
                                                    "mime_type": mime_type,
                                                    "data": data
                                                }
                                            })
                        elif isinstance(tool_content, str):
                            parts.append({"text": tool_content})

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # Merge consecutive same-role messages (Gemini requires alternating user/model)
        merged = []
        for entry in contents:
            if merged and merged[-1]["role"] == entry["role"]:
                merged[-1]["parts"].extend(entry["parts"])
            else:
                merged.append(entry)

        request_body = {"contents": merged}
        if system_instruction:
            request_body["system_instruction"] = system_instruction

        # generationConfig
        gen_config = {}
        if "max_tokens" in anthropic_request:
            gen_config["max_output_tokens"] = anthropic_request["max_tokens"]
        if "temperature" in anthropic_request:
            gen_config["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            gen_config["top_p"] = anthropic_request["top_p"]
        if gen_config:
            request_body["generationConfig"] = gen_config

        return request_body

    def translate_native_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Gemini native generateContent response to Anthropic format."""
        import uuid

        candidates = gemini_response.get("candidates", [])
        content_blocks = []
        stop_reason = "end_turn"

        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    content_blocks.append({"type": "text", "text": part["text"]})

            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "STOP":
                stop_reason = "end_turn"
            elif finish_reason == "MAX_TOKENS":
                stop_reason = "max_tokens"

        usage_metadata = gemini_response.get("usageMetadata", {})

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": self.model_name,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage_metadata.get("promptTokenCount", 0),
                "output_tokens": usage_metadata.get("candidatesTokenCount", 0)
            }
        }


# Import the translation function we need
from .translation import anthropic_to_openai_request