"""
Tests for inline model routing feature.

Tests the ability to dynamically change provider associations for model IDs
via the routing API, as used by the inline dropdown in the dashboard.
"""

import pytest
import os
import database as db


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    """Set up a fresh test database for each test."""
    db_file = os.path.join(tmp_path, "test_inline_routing.db")
    db.DB_PATH = db_file
    db.init_db()
    yield
    db.DB_PATH = "proxy.db"


class TestModelMappingDatabase:
    """Tests for model mapping database operations."""

    def test_add_model_mapping(self):
        """Adding a model mapping stores it correctly."""
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        providers = db.get_providers()
        provider_id = providers[0]["id"]

        db.add_model_mapping("test-model", provider_id)
        mappings = db.get_model_mappings()

        assert len(mappings) == 1
        assert mappings[0]["model_id"] == "test-model"
        assert mappings[0]["provider_id"] == provider_id
        assert mappings[0]["provider_name"] == "Provider1"

    def test_update_model_mapping_provider(self):
        """Updating a model mapping changes the associated provider."""
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=0)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        # Initial mapping to Provider1
        db.add_model_mapping("test-model", p1_id)
        mappings = db.get_model_mappings()
        assert mappings[0]["provider_id"] == p1_id

        # Update to Provider2 (INSERT OR REPLACE)
        db.add_model_mapping("test-model", p2_id)
        mappings = db.get_model_mappings()

        assert len(mappings) == 1  # Should still be one mapping
        assert mappings[0]["provider_id"] == p2_id
        assert mappings[0]["provider_name"] == "Provider2"

    def test_delete_model_mapping(self):
        """Deleting a model mapping removes it correctly."""
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        providers = db.get_providers()
        provider_id = providers[0]["id"]

        db.add_model_mapping("test-model", provider_id)
        assert len(db.get_model_mappings()) == 1

        db.delete_model_mapping("test-model")
        assert len(db.get_model_mappings()) == 0

    def test_multiple_model_mappings(self):
        """Multiple model mappings can coexist."""
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=0)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        db.add_model_mapping("model-a", p1_id)
        db.add_model_mapping("model-b", p2_id)
        db.add_model_mapping("model-c", p1_id)

        mappings = db.get_model_mappings()
        assert len(mappings) == 3

        model_a = next(m for m in mappings if m["model_id"] == "model-a")
        model_b = next(m for m in mappings if m["model_id"] == "model-b")
        assert model_a["provider_id"] == p1_id
        assert model_b["provider_id"] == p2_id


class TestRoutingAPI:
    """Tests for the routing API endpoints."""

    def test_add_mapping_via_api(self):
        """POST /api/routing adds a new mapping."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "model-a", is_active=0)
        providers = db.get_providers()
        provider_id = providers[0]["id"]

        response = client.post("/api/routing", json={"model_id": "my-model", "provider_id": provider_id})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_update_mapping_via_api(self):
        """POST /api/routing updates an existing mapping."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=0)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        # Create initial mapping
        client.post("/api/routing", json={"model_id": "test-model", "provider_id": p1_id})

        # Update to different provider
        response = client.post("/api/routing", json={"model_id": "test-model", "provider_id": p2_id})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Verify the update
        response = client.get("/api/routing")
        mappings = response.json()
        assert len(mappings) == 1
        assert mappings[0]["provider_id"] == p2_id

    def test_get_mappings_returns_provider_id(self):
        """GET /api/routing returns provider_id for each mapping."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "model-a", is_active=0)
        providers = db.get_providers()
        provider_id = providers[0]["id"]

        client.post("/api/routing", json={"model_id": "test-model", "provider_id": provider_id})

        response = client.get("/api/routing")
        mappings = response.json()
        assert len(mappings) == 1
        assert "provider_id" in mappings[0]
        assert mappings[0]["provider_id"] == provider_id
        assert "provider_name" in mappings[0]
        assert mappings[0]["provider_name"] == "TestProvider"

    def test_delete_mapping_via_api(self):
        """DELETE /api/routing/{model_id} removes a mapping."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        db.add_provider("TestProvider", "openai", "http://test.com", "sk-test", "model-a", is_active=0)
        providers = db.get_providers()
        provider_id = providers[0]["id"]

        client.post("/api/routing", json={"model_id": "test-model", "provider_id": provider_id})
        response = client.delete("/api/routing/test-model")
        assert response.status_code == 200

        response = client.get("/api/routing")
        assert response.json() == []

    def test_missing_fields_returns_400(self):
        """POST /api/routing returns 400 if model_id or provider_id missing."""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)

        response = client.post("/api/routing", json={"model_id": "test-model"})
        assert response.status_code == 400

        response = client.post("/api/routing", json={"provider_id": 1})
        assert response.status_code == 400


class TestProviderServiceModelLookup:
    """Tests for provider service model lookup after update."""

    def test_lookup_after_provider_change(self):
        """Provider lookup returns correct provider after mapping update."""
        from core.providers.service import ProviderService

        db.add_provider("Provider1", "openai", "http://test1.com", "key1", "model1", is_active=0)
        db.add_provider("Provider2", "openai", "http://test2.com", "key2", "model2", is_active=0)
        providers = db.get_providers()
        p1_id = providers[0]["id"]
        p2_id = providers[1]["id"]

        db.add_model_mapping("dynamic-model", p1_id)

        service = ProviderService()
        provider = service.get_provider_by_model("dynamic-model")
        assert provider is not None
        assert provider.name == "Provider1"

        # Simulate inline update
        db.add_model_mapping("dynamic-model", p2_id)

        # Need fresh service to avoid cache
        service2 = ProviderService()
        provider = service2.get_provider_by_model("dynamic-model")
        assert provider is not None
        assert provider.name == "Provider2"
