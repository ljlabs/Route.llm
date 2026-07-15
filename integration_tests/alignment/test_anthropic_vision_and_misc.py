"""
Conformance tests for Anthropic image content blocks and the
/v1/messages/count_tokens endpoint.
"""
import pytest
from .validators import validate_anthropic_message

pytestmark = pytest.mark.anthropic

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBA"
    "scY42YAAAAASUVORK5CYII="
)


@pytest.mark.vision
def test_image_content_block_base64(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": TINY_PNG_B64,
                }},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        }],
    })
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


@pytest.mark.vision
def test_image_content_block_url(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages", headers=anthropic_headers, json={
        "model": anthropic_model,
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "url",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg",
                }},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        }],
    })
    # Some routers may not support fetching remote URLs server-side; that's
    # a legitimate 400, but not a 5xx.
    assert resp.status_code < 500, resp.text
    if resp.status_code == 200:
        validate_anthropic_message(resp.json())


def test_count_tokens_endpoint(anthropic_session, anthropic_base_url, anthropic_headers, anthropic_model):
    resp = anthropic_session.post(f"{anthropic_base_url}/v1/messages/count_tokens", headers=anthropic_headers, json={
        "model": anthropic_model,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    })
    if resp.status_code == 404:
        pytest.skip("router does not implement /v1/messages/count_tokens (optional endpoint)")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "input_tokens" in body, f"expected 'input_tokens' in count_tokens response, got {body!r}"
    assert isinstance(body["input_tokens"], int)
