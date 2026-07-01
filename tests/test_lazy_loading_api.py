"""
Tests for the new lazy-loading API endpoints.

Tests /api/logs/metadata and /api/logs/{id}/events endpoints.
"""

import os
import sys
import pytest

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db
from fastapi.testclient import TestClient
from main import app


@pytest.mark.anyio
async def test_logs_metadata_api_endpoint(tmp_path):
    """Test the /api/logs/metadata endpoint returns metadata without events."""
    db_file = os.path.join(tmp_path, "test_logs_metadata_api.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    client = TestClient(app)

    # GET /api/logs/metadata - empty
    response = client.get("/api/logs/metadata")
    assert response.status_code == 200
    assert response.json() == []

    # Add a log with events
    request_id = db.start_request_log(
        provider_name="TestProvider",
        request_method="POST",
        request_path="/v1/messages",
        request_body='{"messages":[]}',
    )
    
    db.add_log_event(
        request_id=request_id,
        stage="provider_request",
        body='{"model":"claude"}',
        status_code=None,
    )
    
    db.complete_request_log(
        request_id=request_id,
        response_status=200,
        response_body='{"content":"response"}',
    )

    # GET /api/logs/metadata - should return metadata without events
    response = client.get("/api/logs/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert len(metadata) == 1
    
    log = metadata[0]
    assert log["provider_name"] == "TestProvider"
    assert log["request_method"] == "POST"
    assert log["request_path"] == "/v1/messages"
    assert log["response_status"] == 200
    
    # Should NOT have events array
    assert "events" not in log, "Metadata endpoint should not include events"


@pytest.mark.anyio
async def test_log_events_api_endpoint(tmp_path):
    """Test the /api/logs/{id}/events endpoint returns events for a specific log."""
    db_file = os.path.join(tmp_path, "test_log_events_api.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    client = TestClient(app)

    # Create a log with events
    request_id = db.start_request_log(
        provider_name="TestProvider",
        request_method="POST",
        request_path="/v1/messages",
        request_body='{"messages":[{"role":"user","content":"Hi"}]}',
    )
    
    db.add_log_event(
        request_id=request_id,
        stage="provider_request",
        body='{"model":"claude"}',
        status_code=None,
    )
    
    db.complete_request_log(
        request_id=request_id,
        response_status=200,
        response_body='{"content":"response"}',
    )

    # GET /api/logs/{id}/events - should return events
    response = client.get(f"/api/logs/{request_id}/events")
    assert response.status_code == 200
    events = response.json()
    assert len(events) == 3  # router_received, provider_request, client_response
    
    # Check stages are correct
    stages = [e["stage"] for e in events]
    assert stages == ["router_received", "provider_request", "client_response"]
    
    # Check each event has expected fields
    for event in events:
        assert "stage" in event
        assert "timestamp" in event
        assert "body" in event
        assert "status_code" in event


@pytest.mark.anyio
async def test_log_events_api_nonexistent(tmp_path):
    """Test the /api/logs/{id}/events endpoint with non-existent log ID."""
    db_file = os.path.join(tmp_path, "test_log_events_nonexistent.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    client = TestClient(app)

    # GET /api/logs/9999/events - should return empty list
    response = client.get("/api/logs/9999/events")
    assert response.status_code == 200
    events = response.json()
    assert events == []


@pytest.mark.anyio
async def test_lazy_loading_workflow(tmp_path):
    """Test the complete lazy-loading workflow: metadata first, then events."""
    db_file = os.path.join(tmp_path, "test_lazy_workflow.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    client = TestClient(app)

    # Create multiple logs
    log_ids = []
    for i in range(3):
        request_id = db.start_request_log(
            provider_name=f"Provider{i}",
            request_method="POST",
            request_path="/v1/messages",
            request_body='{"messages":[]}',
        )
        
        db.add_log_event(
            request_id=request_id,
            stage="provider_request",
            body='{"model":"claude"}',
            status_code=None,
        )
        
        db.complete_request_log(
            request_id=request_id,
            response_status=200,
            response_body='{"content":"response"}',
        )
        
        log_ids.append(request_id)

    # Step 1: Load metadata (fast)
    response = client.get("/api/logs/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert len(metadata) == 3
    
    # Verify metadata structure (no events)
    for meta in metadata:
        assert "id" in meta
        assert "provider_name" in meta
        assert "events" not in meta

    # Step 2: Load events for specific log on demand
    first_log_id = log_ids[0]
    response = client.get(f"/api/logs/{first_log_id}/events")
    assert response.status_code == 200
    events = response.json()
    assert len(events) > 0
    
    # Step 3: Verify we can get another log's events independently
    second_log_id = log_ids[1]
    response = client.get(f"/api/logs/{second_log_id}/events")
    assert response.status_code == 200
    events = response.json()
    assert len(events) > 0


@pytest.mark.anyio
async def test_metadata_vs_full_logs_endpoints(tmp_path):
    """Test that /api/logs/metadata is different from /api/logs."""
    db_file = os.path.join(tmp_path, "test_metadata_vs_full.db")
    db.DB_PATH = db_file
    db.init_db()
    db.clear_logs()

    client = TestClient(app)

    # Create a log with events
    request_id = db.start_request_log(
        provider_name="TestProvider",
        request_method="POST",
        request_path="/v1/messages",
        request_body='{"messages":[]}',
    )
    
    db.add_log_event(
        request_id=request_id,
        stage="provider_request",
        body='{"model":"claude"}',
        status_code=None,
    )
    
    db.complete_request_log(
        request_id=request_id,
        response_status=200,
        response_body='{"content":"response"}',
    )

    # Get full logs
    response = client.get("/api/logs")
    assert response.status_code == 200
    full_logs = response.json()
    assert len(full_logs) == 1
    full_log = full_logs[0]
    assert "events" in full_log
    assert len(full_log["events"]) > 0

    # Get metadata only
    response = client.get("/api/logs/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert len(metadata) == 1
    meta_log = metadata[0]
    assert "events" not in meta_log

    # Verify metadata fields match full log's basic fields
    assert meta_log["id"] == full_log["id"]
    assert meta_log["provider_name"] == full_log["provider_name"]
    assert meta_log["response_status"] == full_log["response_status"]
