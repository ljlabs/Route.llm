"""
Unit tests for the 4-stage request lifecycle logging.

Tests that start_request_log, add_log_event, complete_request_log,
and get_logs all work together correctly.
"""

import os
import sys
import sqlite3
import pytest

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point DB_PATH at a temporary file and initialise it."""
    test_db = str(tmp_path / "test_proxy.db")
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db()
    yield test_db


class TestLogEventsTableExists:
    def test_log_events_table_created(self, fresh_db):
        conn = sqlite3.connect(fresh_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='log_events'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "log_events table should exist after init_db()"


class TestStartRequestLog:
    def test_returns_request_id(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        assert request_id is not None
        assert isinstance(request_id, int)
        assert request_id > 0

    def test_creates_log_row_with_status_zero(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/api/chat",
            request_body='{"messages":[{"role":"user","content":"hi"}]}',
        )
        logs = db.get_logs()
        assert len(logs) == 1
        log = logs[0]
        assert log["id"] == request_id
        assert log["response_status"] == 0
        assert log["response_body"] == ""
        assert log["provider_name"] == "TestProvider"
        assert log["request_method"] == "POST"
        assert log["request_path"] == "/api/chat"

    def test_creates_router_received_event(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/api/chat",
            request_body='{"test":"body"}',
        )
        logs = db.get_logs()
        assert len(logs) == 1
        events = logs[0]["events"]
        assert len(events) == 1
        assert events[0]["stage"] == "router_received"
        assert events[0]["body"] == '{"test":"body"}'
        assert events[0]["status_code"] is None

    def test_returns_none_when_logging_disabled(self):
        db.set_log_limit(-1)
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/api/chat",
            request_body="{}",
        )
        assert request_id is None


class TestAddLogEvent:
    def test_adds_provider_request_event(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/messages",
            request_body='{"original":"request"}',
        )
        db.add_log_event(request_id, stage="provider_request", body='{"wrapped":"request"}')

        logs = db.get_logs()
        events = logs[0]["events"]
        assert len(events) == 2
        assert events[0]["stage"] == "router_received"
        assert events[1]["stage"] == "provider_request"
        assert events[1]["body"] == '{"wrapped":"request"}'
        assert events[1]["status_code"] is None

    def test_adds_provider_response_event_with_status(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/messages",
            request_body="{}",
        )
        db.add_log_event(request_id, stage="provider_request", body='{"req":"body"}')
        db.add_log_event(
            request_id,
            stage="provider_response",
            body='{"id":"resp-123","choices":[]}',
            status_code=200,
        )

        logs = db.get_logs()
        events = logs[0]["events"]
        assert len(events) == 3
        assert events[2]["stage"] == "provider_response"
        assert events[2]["status_code"] == 200

    def test_noop_when_request_id_is_none(self):
        # Should not raise
        db.add_log_event(None, stage="provider_request", body="something")
        logs = db.get_logs()
        assert len(logs) == 0


class TestCompleteRequestLog:
    def test_updates_log_row_and_adds_client_response_event(self):
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/api/chat",
            request_body='{"msg":"hello"}',
        )
        db.add_log_event(request_id, stage="provider_request", body='{"wrapped":"data"}')
        db.add_log_event(request_id, stage="provider_response", body='{"resp":"data"}', status_code=200)
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"final":"response"}',
            tokens_sent=10,
            tokens_received=20,
            latency_ms=150,
        )

        logs = db.get_logs()
        assert len(logs) == 1
        log = logs[0]

        # Verify the log row was updated (not overwritten/duplicated)
        assert log["id"] == request_id
        assert log["response_status"] == 200
        assert log["response_body"] == '{"final":"response"}'
        assert log["tokens_sent"] == 10
        assert log["tokens_received"] == 20
        assert log["latency_ms"] == 150

        # Verify all 4 events exist in order
        events = log["events"]
        assert len(events) == 4
        assert events[0]["stage"] == "router_received"
        assert events[1]["stage"] == "provider_request"
        assert events[2]["stage"] == "provider_response"
        assert events[3]["stage"] == "client_response"
        assert events[3]["status_code"] == 200
        assert events[3]["body"] == '{"final":"response"}'

    def test_does_not_create_duplicate_log_row(self):
        """Ensure complete_request_log UPDATES the existing row, doesn't INSERT a new one."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/api/chat",
            request_body="{}",
        )
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"done":true}',
        )
        logs = db.get_logs()
        assert len(logs) == 1, f"Expected 1 log row, got {len(logs)}"
        assert logs[0]["id"] == request_id

    def test_noop_when_request_id_is_none(self):
        db.complete_request_log(
            request_id=None,
            response_status=200,
            response_body="{}",
        )
        logs = db.get_logs()
        assert len(logs) == 0


class TestFullLifecycleFlow:
    """End-to-end test simulating what the router does."""

    def test_full_4_stage_lifecycle(self):
        # Stage 1 — request arrives at router
        request_id = db.start_request_log(
            provider_name="OpenRouter",
            request_method="POST",
            request_path="/api/chat",
            request_body='{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}',
        )
        assert request_id is not None

        # At this point the log should be visible with status 0
        logs = db.get_logs()
        assert len(logs) == 1
        assert logs[0]["response_status"] == 0
        assert len(logs[0]["events"]) == 1

        # Stage 2 — about to send to provider
        db.add_log_event(
            request_id,
            stage="provider_request",
            body='{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"max_tokens":4096}',
        )

        logs = db.get_logs()
        assert len(logs[0]["events"]) == 2

        # Stage 3 — response received from provider
        db.add_log_event(
            request_id,
            stage="provider_response",
            body='{"id":"chatcmpl-123","choices":[{"message":{"content":"Hello!"}}]}',
            status_code=200,
        )

        logs = db.get_logs()
        assert len(logs[0]["events"]) == 3

        # Stage 4 — response sent to client
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"id":"chatcmpl-123","choices":[{"message":{"content":"Hello!"}}]}',
            tokens_sent=15,
            tokens_received=5,
            latency_ms=320,
        )

        logs = db.get_logs()
        assert len(logs) == 1  # Still just 1 row
        log = logs[0]
        assert log["response_status"] == 200
        assert log["latency_ms"] == 320
        events = log["events"]
        assert len(events) == 4
        stages = [e["stage"] for e in events]
        assert stages == ["router_received", "provider_request", "provider_response", "client_response"]

    def test_lifecycle_with_error_response(self):
        """Provider returns 401 — should still get all 4 stages."""
        request_id = db.start_request_log(
            provider_name="Anthropic",
            request_method="POST",
            request_path="/v1/messages",
            request_body='{"model":"claude-3","messages":[]}',
        )
        db.add_log_event(request_id, stage="provider_request", body='{"wrapped":"body"}')
        db.add_log_event(
            request_id,
            stage="provider_response",
            body='{"error":{"message":"Invalid API Key"}}',
            status_code=401,
        )
        db.complete_request_log(
            request_id=request_id,
            response_status=401,
            response_body='{"error":{"message":"Invalid API Key"}}',
        )

        logs = db.get_logs()
        log = logs[0]
        assert log["response_status"] == 401
        events = log["events"]
        assert len(events) == 4
        assert events[2]["status_code"] == 401
        assert events[3]["status_code"] == 401

    def test_multiple_requests_dont_interfere(self):
        """Two concurrent requests each get their own events."""
        id1 = db.start_request_log("ProvA", "POST", "/api/chat", '{"req":1}')
        id2 = db.start_request_log("ProvB", "POST", "/api/chat", '{"req":2}')

        db.add_log_event(id1, "provider_request", body="req1_to_provider")
        db.add_log_event(id2, "provider_request", body="req2_to_provider")
        db.add_log_event(id2, "provider_response", body="resp2_from_provider", status_code=200)
        db.add_log_event(id1, "provider_response", body="resp1_from_provider", status_code=200)

        db.complete_request_log(id2, 200, "final2")
        db.complete_request_log(id1, 200, "final1")

        logs = db.get_logs()
        assert len(logs) == 2

        # Logs are DESC by id, so id2 first
        log2 = next(l for l in logs if l["id"] == id2)
        log1 = next(l for l in logs if l["id"] == id1)

        assert len(log1["events"]) == 4
        assert len(log2["events"]) == 4
        assert log1["events"][0]["body"] == '{"req":1}'
        assert log2["events"][0]["body"] == '{"req":2}'


class TestGetLogsReturnsEvents:
    def test_events_key_always_present(self):
        """Even legacy logs (via add_log) should have an events key (empty list)."""
        db.add_log(
            provider_name="Legacy",
            request_method="GET",
            request_path="/old",
            request_body="{}",
            response_status=200,
            response_body="{}",
        )
        logs = db.get_logs()
        assert len(logs) == 1
        assert "events" in logs[0]
        assert logs[0]["events"] == []

    def test_events_ordered_by_insertion(self):
        request_id = db.start_request_log("P", "POST", "/x", "{}")
        db.add_log_event(request_id, "provider_request", body="a")
        db.add_log_event(request_id, "provider_response", body="b", status_code=200)
        db.complete_request_log(request_id, 200, "c")

        logs = db.get_logs()
        stages = [e["stage"] for e in logs[0]["events"]]
        assert stages == ["router_received", "provider_request", "provider_response", "client_response"]


class TestEnforceLimitCascadesEvents:
    def test_deleting_old_logs_removes_their_events(self, fresh_db):
        db.set_log_limit(2)

        id1 = db.start_request_log("P", "POST", "/x", "{}")
        db.add_log_event(id1, "provider_request")
        db.complete_request_log(id1, 200, "resp1")

        id2 = db.start_request_log("P", "POST", "/x", "{}")
        db.complete_request_log(id2, 200, "resp2")

        id3 = db.start_request_log("P", "POST", "/x", "{}")
        db.complete_request_log(id3, 200, "resp3")

        # Limit is 2, so id1 should be pruned
        logs = db.get_logs()
        assert len(logs) == 2
        ids = {l["id"] for l in logs}
        assert id1 not in ids

        # Verify events for id1 are gone
        conn = sqlite3.connect(fresh_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log_events WHERE request_id = ?", (id1,))
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0, "Events for pruned log should be cascade-deleted"
