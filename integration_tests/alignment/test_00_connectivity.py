"""
Fast pre-flight check, meant to run before the rest of the suite
(pytest collects files alphabetically, and '00' sorts first).

If your server isn't up, misconfigured, or on a different port, you
want ONE fast, clear failure here -- not 80 tests each hanging for
30s before timing out.
"""
import socket
from urllib.parse import urlparse

import pytest
import requests


def _tcp_reachable(base_url, timeout=3):
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except OSError as e:
        return False, str(e)


def test_openai_server_is_reachable(openai_base_url):
    ok, err = _tcp_reachable(openai_base_url)
    assert ok, (
        f"Could not open a TCP connection to {openai_base_url} ({err}). "
        f"Is the router actually listening there? Check OPENAI_BASE_URL / "
        f"the port your server binds to, and any firewall in between."
    )


def test_openai_server_responds_to_http(openai_base_url, openai_headers, openai_model):
    try:
        resp = requests.post(
            f"{openai_base_url}/v1/chat/completions",
            headers=openai_headers,
            json={"model": openai_model, "messages": [{"role": "user", "content": "ping"}]},
            timeout=10,
        )
    except requests.exceptions.Timeout:
        pytest.fail(
            f"TCP connected to {openai_base_url} but got no HTTP response within 10s. "
            f"The port is open but nothing is answering -- check that your app (not just "
            f"the socket/listener) is actually running and isn't hung/deadlocked."
        )
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Connection to {openai_base_url} was refused/reset: {e}")
    assert resp.status_code < 500, f"server responded but with a 5xx: {resp.status_code}: {resp.text[:500]}"


def test_anthropic_server_is_reachable(anthropic_base_url):
    ok, err = _tcp_reachable(anthropic_base_url)
    assert ok, (
        f"Could not open a TCP connection to {anthropic_base_url} ({err}). "
        f"Is the router actually listening there? Check ANTHROPIC_BASE_URL / "
        f"the port your server binds to, and any firewall in between."
    )


def test_anthropic_server_responds_to_http(anthropic_base_url, anthropic_headers, anthropic_model):
    try:
        resp = requests.post(
            f"{anthropic_base_url}/v1/messages",
            headers=anthropic_headers,
            json={"model": anthropic_model, "max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]},
            timeout=10,
        )
    except requests.exceptions.Timeout:
        pytest.fail(
            f"TCP connected to {anthropic_base_url} but got no HTTP response within 10s. "
            f"The port is open but nothing is answering -- check that your app (not just "
            f"the socket/listener) is actually running and isn't hung/deadlocked."
        )
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Connection to {anthropic_base_url} was refused/reset: {e}")
    assert resp.status_code < 500, f"server responded but with a 5xx: {resp.status_code}: {resp.text[:500]}"