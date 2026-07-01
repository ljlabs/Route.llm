"""
Tests for lazy-loading logs and metrics optimization.

Tests that logs metadata can be loaded separately from events,
and that events can be fetched on-demand per log.
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


class TestGetLogsMetadata:
    """Test the new get_logs_metadata() function for fast metadata-only loading."""
    
    def test_returns_list(self):
        """get_logs_metadata should return a list even when empty."""
        metadata = db.get_logs_metadata()
        assert isinstance(metadata, list)
        assert len(metadata) == 0
    
    def test_includes_required_fields(self):
        """Metadata should include essential fields without events."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        # Simulate a response
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        metadata = db.get_logs_metadata()
        assert len(metadata) == 1
        
        log = metadata[0]
        assert "id" in log
        assert "timestamp" in log
        assert "provider_name" in log
        assert "request_method" in log
        assert "request_path" in log
        assert "response_status" in log
        assert "latency_ms" in log
        assert "tokens_sent" in log
        assert "tokens_received" in log
    
    def test_does_not_include_events(self):
        """Metadata should NOT include events array for fast loading."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        metadata = db.get_logs_metadata()
        assert len(metadata) == 1
        
        log = metadata[0]
        assert "events" not in log, "Events should not be included in metadata for performance"


class TestGetLogEvents:
    """Test the new get_log_events() function for on-demand event fetching."""
    
    def test_returns_empty_list_for_nonexistent_log(self):
        """get_log_events should return empty list for non-existent log ID."""
        events = db.get_log_events(9999)
        assert isinstance(events, list)
        assert len(events) == 0
    
    def test_returns_events_for_existing_log(self):
        """get_log_events should return all events for a specific log."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        # Add a provider_request event
        db.add_log_event(
            request_id=request_id,
            stage="provider_request",
            body='{"model":"claude-3"}',
            status_code=None,
        )
        
        # Complete the request
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        events = db.get_log_events(request_id)
        assert len(events) == 3  # router_received, provider_request, client_response
        
        # Check stages are present
        stages = [e["stage"] for e in events]
        assert "router_received" in stages
        assert "provider_request" in stages
        assert "client_response" in stages
    
    def test_events_ordered_by_timestamp(self):
        """Events should be returned in chronological order."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        db.add_log_event(
            request_id=request_id,
            stage="provider_request",
            body='{"model":"claude-3"}',
            status_code=None,
        )
        
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        events = db.get_log_events(request_id)
        
        # Verify order by checking stages
        expected_order = ["router_received", "provider_request", "client_response"]
        actual_order = [e["stage"] for e in events]
        assert actual_order == expected_order


class TestMetadataVsFullLogs:
    """Test that metadata and full logs are complementary."""
    
    def test_metadata_is_faster_than_full_logs(self):
        """Metadata queries should be significantly faster (no event joins)."""
        import time
        
        # Create multiple logs with events
        for i in range(10):
            request_id = db.start_request_log(
                provider_name=f"Provider{i % 3}",
                request_method="POST",
                request_path="/v1/chat/completions",
                request_body='{"messages":[]}',
            )
            
            db.add_log_event(
                request_id=request_id,
                stage="provider_request",
                body='{"model":"claude-3"}',
                status_code=None,
            )
            
            db.complete_request_log(
                request_id=request_id,
                response_status=200,
                response_body='{"content":"response"}',
            )
        
        # Time metadata load
        start = time.time()
        metadata = db.get_logs_metadata()
        metadata_time = time.time() - start
        
        # Time full logs load
        start = time.time()
        full_logs = db.get_logs()
        full_time = time.time() - start
        
        assert len(metadata) == 10
        assert len(full_logs) == 10
        
        # Metadata should be faster (at least within same order of magnitude)
        # In practice, metadata should be 2-5x faster depending on event count
        print(f"\nMetadata load time: {metadata_time:.6f}s")
        print(f"Full logs load time: {full_time:.6f}s")
        assert metadata_time <= full_time, "Metadata should be faster or equal"
    
    def test_combining_metadata_and_events_recreates_full_log(self):
        """Combining metadata with fetched events should recreate a full log."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        db.add_log_event(
            request_id=request_id,
            stage="provider_request",
            body='{"model":"claude-3"}',
            status_code=None,
        )
        
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        # Get metadata and events separately
        metadata = db.get_logs_metadata()
        meta_log = metadata[0]
        
        events = db.get_log_events(request_id)
        
        # Get full log for comparison
        full_logs = db.get_logs()
        full_log = full_logs[0]
        
        # Metadata fields should match full log
        assert meta_log["id"] == full_log["id"]
        assert meta_log["timestamp"] == full_log["timestamp"]
        assert meta_log["provider_name"] == full_log["provider_name"]
        assert meta_log["response_status"] == full_log["response_status"]
        
        # Events from separate call should match full log events
        assert len(events) == len(full_log["events"])
        for i, event in enumerate(events):
            assert event["stage"] == full_log["events"][i]["stage"]
            assert event["status_code"] == full_log["events"][i]["status_code"]


class TestBackwardCompatibility:
    """Test that the original get_logs() function still works."""
    
    def test_get_logs_still_includes_events(self):
        """The original get_logs() should still return full logs with events."""
        request_id = db.start_request_log(
            provider_name="TestProvider",
            request_method="POST",
            request_path="/v1/chat/completions",
            request_body='{"messages":[]}',
        )
        
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        logs = db.get_logs()
        assert len(logs) == 1
        log = logs[0]
        
        # Should have events array
        assert "events" in log
        assert isinstance(log["events"], list)
        assert len(log["events"]) > 0
