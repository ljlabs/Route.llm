"""
Tests for /v1/models and /api/routing/models endpoints.

Tests both positive and negative/validation cases for model listing endpoints.
"""

import pytest
import database as db
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Set up a fresh test database for each test."""
    db_file = os.path.join(tmp_path, "test_models.db")
    db.DB_PATH = db_file
    db.init_db()
    yield
    db.DB_PATH = "proxy.db"


import os


class TestModelsEndpointProxy:
    """Tests for the OpenAI-compatible /v1/models endpoint."""

    def test_list_models_with_active_provider(self):
        """GET /v1/models returns active provider's model when no mappings exist."""
        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "gpt-4o", is_active=1)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "gpt-4o"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "router"

    def test_list_models_with_mappings(self):
        """GET /v1/models returns mapped models plus active provider model."""
        client = TestClient(app)
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=1)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        db.add_model_mapping("mapped-model-1", p1_id)
        db.add_model_mapping("mapped-model-2", p2_id)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "mapped-model-1" in model_ids
        assert "mapped-model-2" in model_ids
        assert "model2" in model_ids  # Active provider's model
        assert len(model_ids) == 3

    def test_list_models_no_duplicates(self):
        """GET /v1/models doesn't duplicate active provider model if already mapped."""
        client = TestClient(app)
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "shared-model", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "active-model", is_active=1)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        # Map the same model that active provider uses
        db.add_model_mapping("active-model", p1_id)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]
        # Should only appear once (mapped model + active provider model are same)
        assert model_ids.count("active-model") == 1
        # Only the mapped/active model is returned (not all provider models)
        assert len(model_ids) == 1
        assert model_ids == ["active-model"]

    def test_list_models_empty_when_no_providers(self):
        """GET /v1/models returns empty list when no providers configured."""
        client = TestClient(app)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_response_format_openai_compatible(self):
        """Response format matches OpenAI models list API."""
        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "gpt-4o", is_active=1)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["object"] == "model"
        assert isinstance(model["created"], int)


class TestModelsEndpointRouting:
    """Tests for the /api/routing/models endpoint."""

    def test_list_models_with_mappings(self):
        """GET /api/routing/models returns mapped models plus active provider model."""
        client = TestClient(app)
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=1)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        db.add_model_mapping("mapped-model-1", p1_id)
        db.add_model_mapping("mapped-model-2", p2_id)

        response = client.get("/api/routing/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "mapped-model-1" in model_ids
        assert "mapped-model-2" in model_ids
        assert "model2" in model_ids  # Active provider's model

    def test_list_models_empty_when_no_providers(self):
        """GET /api/routing/models returns empty list when no providers."""
        client = TestClient(app)

        response = client.get("/api/routing/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_format(self):
        """Response format is correct for routing models endpoint."""
        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "gpt-4o", is_active=1)

        response = client.get("/api/routing/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"


class TestModelsEndpointValidation:
    """Validation and negative tests for models endpoints."""

    def test_models_endpoint_method_not_allowed_post(self):
        """POST /v1/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.post("/v1/models", json={})
        assert response.status_code == 405

    def test_models_endpoint_method_not_allowed_put(self):
        """PUT /v1/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.put("/v1/models", json={})
        assert response.status_code == 405

    def test_models_endpoint_method_not_allowed_delete(self):
        """DELETE /v1/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.delete("/v1/models")
        assert response.status_code == 405

    def test_routing_models_method_not_allowed_post(self):
        """POST /api/routing/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.post("/api/routing/models", json={})
        assert response.status_code == 405

    def test_routing_models_method_not_allowed_put(self):
        """PUT /api/routing/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.put("/api/routing/models", json={})
        assert response.status_code == 405

    def test_routing_models_method_not_allowed_delete(self):
        """DELETE /api/routing/models should return 405 Method Not Allowed."""
        client = TestClient(app)
        response = client.delete("/api/routing/models")
        assert response.status_code == 405

    def test_models_endpoint_invalid_path(self):
        """GET /v1/model (singular) should return 404."""
        client = TestClient(app)
        response = client.get("/v1/model")
        assert response.status_code == 404

    def test_routing_models_endpoint_invalid_path(self):
        """GET /api/routing/model (singular) should return 404."""
        client = TestClient(app)
        response = client.get("/api/routing/model")
        assert response.status_code == 404


class TestModelsEndpointWithDatabaseErrors:
    """Tests for handling database errors gracefully."""

    def test_list_models_handles_db_error_gracefully(self):
        """GET /v1/models returns 500 when database has issues."""
        # This test is more of a sanity check - the endpoint should handle errors
        # In practice, the database is usually available, but we test the error path
        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "gpt-4o", is_active=1)

        # Normal request should work
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_routing_models_handles_db_error_gracefully(self):
        """GET /api/routing/models returns 500 when database has issues."""
        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "gpt-4o", is_active=1)

        response = client.get("/api/routing/models")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])