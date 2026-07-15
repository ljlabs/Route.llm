"""Media upload compatibility coverage for the local OpenAI/Anthropic mock."""

from fastapi.testclient import TestClient
import pytest

from core.providers.gemini import GeminiProvider
from core.providers.translation import anthropic_to_openai_request, openai_to_anthropic_request
from load_test import mock_server

TINY_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAscY42YAAAAASUVORK5CYII="
TINY_PDF = "JVBERi0xLjQKJUVPRgo="
TINY_TEXT = "SGVsbG8gZmlsZSE="


@pytest.fixture(autouse=True)
def reset_mock_state():
    original_latency = mock_server.response_latency_ms
    mock_server.response_latency_ms = 0
    for key in ("total_image_requests", "pdf_requests", "generic_file_requests"):
        mock_server._stats[key] = 0
    yield
    mock_server.response_latency_ms = original_latency


def test_openai_chat_mock_accepts_base64_image_url():
    response = TestClient(mock_server.app).post("/v1/chat/completions", json={
        "model": "mock-model", "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_IMAGE}"}},
        ]}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "images=1 pdfs=0 files=0" in body["choices"][0]["message"]["content"]


def test_openai_responses_mock_accepts_base64_image_and_file_forms():
    response = TestClient(mock_server.app).post("/v1/responses", json={
        "model": "mock-model", "max_output_tokens": 8, "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{TINY_IMAGE}"},
            {"type": "input_file", "filename": "sample.pdf", "file_data": f"data:application/pdf;base64,{TINY_PDF}"},
            {"type": "input_file", "filename": "notes.txt", "file_data": f"data:text/plain;base64,{TINY_TEXT}"},
        ]}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response" and body["status"] == "completed"
    signal = body["output"][0]["content"][0]["text"]
    assert "images=1 pdfs=1 files=1" in signal
    assert "image/jpeg,application/pdf,text/plain" in signal


def test_anthropic_mock_accepts_base64_image_pdf_and_generic_document():
    response = TestClient(mock_server.app).post("/v1/messages", json={
        "model": "mock-model", "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": TINY_IMAGE}},
            {"type": "document", "title": "sample.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": TINY_PDF}},
            {"type": "document", "title": "notes.txt", "source": {"type": "base64", "media_type": "text/plain", "data": TINY_TEXT}},
        ]}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    signal = body["content"][0]["text"]
    assert "images=1 pdfs=1 files=1" in signal
    assert "image/webp,application/pdf,text/plain" in signal


def test_base64_documents_round_trip_between_anthropic_and_openai():
    original = {"model": "claude", "max_tokens": 8, "messages": [{"role": "user", "content": [
        {"type": "document", "title": "sample.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": TINY_PDF}},
        {"type": "document", "title": "notes.txt", "source": {"type": "base64", "media_type": "text/plain", "data": TINY_TEXT}},
    ]}]}
    openai = anthropic_to_openai_request(original, "gpt-mock")
    files = openai["messages"][0]["content"]
    assert [block["type"] for block in files] == ["file", "file"]
    assert files[0]["file"] == {"filename": "sample.pdf", "file_data": TINY_PDF}
    assert files[1]["file"] == {"filename": "notes.txt", "file_data": TINY_TEXT}
    round_trip = openai_to_anthropic_request(openai, "claude-mock")
    documents = round_trip["messages"][0]["content"]
    assert [block["source"] for block in documents] == [
        {"type": "base64", "media_type": "application/pdf", "data": TINY_PDF},
        {"type": "base64", "media_type": "text/plain", "data": TINY_TEXT},
    ]


def test_openai_input_file_and_input_image_translate_to_anthropic_blocks():
    request = {"model": "gpt", "messages": [{"role": "user", "content": [
        {"type": "input_image", "image_url": f"data:image/png;base64,{TINY_IMAGE}"},
        {"type": "file", "file": {"filename": "notes.txt", "file_data": f"data:text/plain;base64,{TINY_TEXT}"}},
    ]}]}
    translated = openai_to_anthropic_request(request, "claude")
    blocks = translated["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["data"] == TINY_IMAGE
    assert blocks[1]["type"] == "document" and blocks[1]["source"]["data"] == TINY_TEXT


def test_gemini_native_request_preserves_base64_generic_documents():
    provider = GeminiProvider(
        name="Gemini",
        endpoint_url="https://example.test/v1beta/openai/chat/completions",
        api_key="key",
        model_name="gemini-test",
    )
    request = {"messages": [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "text/plain", "data": TINY_TEXT}},
    ]}]}
    assert provider.detect_pdf_content(request) is True
    assert provider.build_native_pdf_request(request)["contents"][0]["parts"] == [
        {"inline_data": {"mime_type": "text/plain", "data": TINY_TEXT}}
    ]


@pytest.fixture
def openai_api_backend(tmp_path):
    """Configure an isolated OpenAI Chat Completions upstream."""
    import database as db

    original_path = db.DB_PATH
    db.DB_PATH = str(tmp_path / "openai_media_api.db")
    db.init_db()
    endpoint = "https://upstream.test/v1/chat/completions"
    db.add_provider(
        "OpenAI media upstream", "openai", endpoint, "test-key", "media-model", is_active=1
    )
    from core.providers.service import get_provider_service
    provider_service = get_provider_service()
    provider_service.reload_active_provider()
    yield endpoint
    db.DB_PATH = original_path
    provider_service._invalidate_cache()


def _openai_upstream_response():
    return {
        "id": "chatcmpl-media",
        "object": "chat.completion",
        "created": 1,
        "model": "media-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "media received"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_openai_chat_api_preserves_url_and_base64_images(openai_api_backend):
    """The public OpenAI endpoint must preserve standard image_url parts upstream."""
    import json
    import httpx
    import respx
    from main import app

    content = [
        {"type": "text", "text": "Compare these images."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_IMAGE}"}},
    ]
    with respx.mock(assert_all_called=True) as mock:
        upstream = mock.post(openai_api_backend).mock(
            return_value=httpx.Response(200, json=_openai_upstream_response())
        )
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "media-model", "messages": [{"role": "user", "content": content}]
            })

    assert response.status_code == 200, response.text
    payload = json.loads(upstream.calls.last.request.content)
    assert payload["messages"][0]["content"] == content


def test_openai_chat_api_preserves_pdf_file_part(openai_api_backend):
    """The public OpenAI endpoint must use the Chat Completions file schema upstream."""
    import json
    import httpx
    import respx
    from main import app

    content = [
        {"type": "text", "text": "Read the PDF."},
        {"type": "file", "file": {"filename": "sample.pdf", "file_data": TINY_PDF}},
    ]
    with respx.mock(assert_all_called=True) as mock:
        upstream = mock.post(openai_api_backend).mock(
            return_value=httpx.Response(200, json=_openai_upstream_response())
        )
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={
                "model": "media-model", "messages": [{"role": "user", "content": content}]
            })

    assert response.status_code == 200, response.text
    payload = json.loads(upstream.calls.last.request.content)
    assert payload["messages"][0]["content"] == content
    assert payload["messages"][0]["content"][1]["type"] == "file"