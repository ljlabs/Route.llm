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
    assert [block["type"] for block in files] == ["input_file", "input_file"]
    assert files[0]["file_data"] == f"data:application/pdf;base64,{TINY_PDF}"
    assert files[1]["file_data"] == f"data:text/plain;base64,{TINY_TEXT}"
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
