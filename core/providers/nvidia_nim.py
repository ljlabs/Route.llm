"""
Nvidia NIM Providers

NvidiaNimChatProvider   — chat completions via NIM's OpenAI-compatible API,
                          with NIM-tuned hyperparameter defaults that the client
                          can override on a per-request basis.
NvidiaNimEmbeddingProvider — embedding requests for NIM endpoints.
"""

from typing import Any, Dict
from .base import BaseProvider
from .openai import OpenAIProvider
from .translation import anthropic_to_openai_request, sanitize_openai_payload
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Default hyperparameters tuned for NIM inference
# ---------------------------------------------------------------------------
_NIM_DEFAULTS: Dict[str, Any] = {
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.12,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.05,
}


class NvidiaNimChatProvider(OpenAIProvider):
    """
    Provider for NVIDIA NIM chat-completion endpoints.

    Wraps requests in OpenAI format and injects NIM-specific hyperparameter
    defaults.  Any parameter already present in the incoming Anthropic request
    takes precedence over the defaults.
    """

    def __init__(self, **kwargs):
        super().__init__(api_type="nvidia_nim", **kwargs)

    def wrap_request(self, anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic request to NIM/OpenAI format with default hyperparams.

        Priority order (highest → lowest):
          1. Value explicitly supplied by the client in the incoming request
          2. NIM default from _NIM_DEFAULTS

        Standard parameters (max_tokens, temperature, top_p, stream) are handled
        by the Anthropic→OpenAI translation layer.  NIM-specific parameters
        (repetition_penalty, frequency_penalty, presence_penalty) are not part of
        the Anthropic spec, so we lift them directly from the raw request here
        before they would otherwise be discarded by the translation step.
        """
        # Collect NIM-specific params the client explicitly provided so we can
        # restore them after the translation (which only knows Anthropic fields).
        _nim_only_keys = ("repetition_penalty", "frequency_penalty", "presence_penalty")
        client_nim_params = {
            k: anthropic_request[k]
            for k in _nim_only_keys
            if k in anthropic_request
        }

        payload = anthropic_to_openai_request(anthropic_request, self.model_name)
        payload = sanitize_openai_payload(payload, is_gemini=False)

        # Apply NIM defaults for any hyperparameter not already present.
        # max_tokens / temperature / top_p are already in the payload when the
        # client set them (the translation layer copied them); the NIM-only params
        # were lifted above and are merged back in here.
        for key, default_value in _NIM_DEFAULTS.items():
            if key in client_nim_params:
                payload[key] = client_nim_params[key]   # client wins
            elif key not in payload:
                payload[key] = default_value             # fall back to NIM default

        return payload


class NvidiaNimEmbeddingProvider(BaseProvider):
    """Provider for Nvidia NIM embedding APIs."""

    def __init__(self, **kwargs):
        kwargs.pop("api_type", None)
        super().__init__(api_type="embedding_nvidia_nim", **kwargs)

    def wrap_request(self, embedding_request: Dict[str, Any]) -> Dict[str, Any]:
        """Build NIM embedding request, forwarding NIM-specific fields."""
        payload = {
            "model": self.model_name,
            "input": embedding_request.get("input", ""),
        }
        payload["input_type"] = embedding_request.get("input_type", "query")
        for key in ("encoding_format", "truncate"):
            if key in embedding_request:
                payload[key] = embedding_request[key]
        return payload

    def unwrap_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        return provider_response

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_stream_translator(self, target_format: str = "anthropic"):
        return None

    def requires_translation(self) -> bool:
        return False
