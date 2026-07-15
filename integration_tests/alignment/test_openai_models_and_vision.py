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


@pytest.mark.vision
def test_pdf_file_content_part(openai_session, openai_base_url, openai_headers, openai_model):
    """Send a PDF through the real OpenAI Chat Completions request schema."""
    pdf_b64 = (
        "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIg"
        "MCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBv"
        "YmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXSAvUmVz"
        "b3VyY2VzIDw8IC9Gb250IDw8IC9GMSA0IDAgUiA+PiA+PiAvQ29udGVudHMgNSAwIFIgPj4KZW5kb2Jq"
        "CjQgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNh"
        "ID4+CmVuZG9iago1IDAgb2JqCjw8IC9MZW5ndGggNTAgPj4Kc3RyZWFtCkJUIC9GMSAxOCBUZiA3MiA3"
        "MjAgVGQgKFJPVVRFUl9QREZfU0VOVElORUwpIFRqIEVUCmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDYK"
        "MDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAK"
        "MDAwMDAwMDExNSAwMDAwMCBuIAowMDAwMDAwMjQxIDAwMDAwIG4gCjAwMDAwMDAzMTEgMDAwMDAgbiAK"
        "dHJhaWxlcgo8PCAvU2l6ZSA2IC9Sb290IDEgMCBSID4+CnN0YXJ0eHJlZgo0MTEKJSVFT0YK"
    )
    resp = openai_session.post(f"{openai_base_url}/v1/chat/completions", headers=openai_headers, json={
        "model": openai_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Return the exact identifier written in this PDF."},
                {"type": "file", "file": {"filename": "sentinel.pdf", "file_data": pdf_b64}},
            ],
        }],
    })
    assert resp.status_code == 200, resp.text
    validate_openai_chat_completion(resp.json())