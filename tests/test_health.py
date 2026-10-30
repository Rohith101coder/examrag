"""Tests for health and readiness probe endpoints."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.dependencies import get_settings, get_vector_store
from app.core.config import Settings
from app.infrastructure.vector_store import QdrantVectorStore


def test_liveness_endpoint_success(client: TestClient) -> None:
    """Liveness probe must return 200 without checking external services."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data
    assert "environment" in data


def test_readiness_endpoint_success(client: TestClient) -> None:
    """Readiness probe must return 200 when all dependencies are ready."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["dependencies"]["qdrant"]["status"] == "ok"
    assert data["dependencies"]["openrouter"]["status"] == "ok"


def test_readiness_endpoint_qdrant_failure(client: TestClient, mock_vector_store: MagicMock) -> None:
    """Readiness probe must return 503 when Qdrant is unreachable."""
    mock_vector_store.health_check.return_value = False

    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["dependencies"]["qdrant"]["status"] == "error"


def test_readiness_endpoint_missing_openrouter_key(client: TestClient, test_settings: Settings) -> None:
    """Readiness probe must return 503 when OpenRouter key is empty or default."""
    test_settings.OPENROUTER_API_KEY = ""

    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["dependencies"]["openrouter"]["status"] == "error"

