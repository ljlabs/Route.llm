"""
Conformance tests for GET /v1/models, GET /v1/models/{id}, and
image (vision) content parts in chat completions.
"""
import pytest
from .validators import validate_openai_models_list, validate_openai_chat_completion, validate_openai_error

pytestmark = pytest.mark.openai


def test_list_models(openai_session, openai_base_url, openai_headers):
    resp = openai_session.get(f"{openai_base_url}/v1/models", headers=openai_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    validate_openai_models_list(body)
    assert len(body["data"]) >= 1, "expected at least one model listed"


def test_retrieve_single_model(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.get(f"{openai_base_url}/v1/models/{openai_model}", headers=openai_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("id") == openai_model
    assert body.get("object") == "model"


def test_retrieve_unknown_model_404(openai_session, openai_base_url, openai_headers):
    resp = openai_session.get(f"{openai_base_url}/v1/models/definitely-not-real-xyz", headers=openai_headers)
    assert resp.status_code == 404, f"expected 404 for unknown model id, got {resp.status_code}: {resp.text}"
    validate_openai_error(resp.json(), resp.status_code)


@pytest.mark.vision
def test_image_url_content_part(openai_session, openai_base_url, openai_headers, openai_model):
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image? One word."},
                {"type": "image_url", "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg"
                }},
            ],
        }],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())


@pytest.mark.vision
def test_image_base64_content_part(openai_session, openai_base_url, openai_headers, openai_model):
    # 1x1 red pixel PNG, base64-encoded
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBA"
        "scY42YAAAAASUVORK5CYII="
    )
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one word."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{tiny_png_b64}"
                }},
            ],
        }],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())
