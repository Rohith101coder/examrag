"""Pytest fixtures, dependency overrides, and mock objects for testing ExamRAG."""

import io
from typing import Generator
from unittest.mock import MagicMock

import pypdf
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_embedding_service,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore
from app.main import create_application
from app.services.llm_service import LLMService


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        APP_NAME="ExamRAG-Test",
        APP_ENV="test",
        API_V1_PREFIX="/api/v1",
        LOG_LEVEL="DEBUG",
        OPENROUTER_API_KEY="test-mock-openrouter-key",
        QDRANT_URL="https://mock-cluster.cloud.qdrant.io",
        QDRANT_API_KEY="test-mock-qdrant-key",
        QDRANT_COLLECTION_NAME="test_documents",
        INGESTION_API_KEY="test-secret-ingestion-key",
        PUBLIC_CHAT_ENABLED=True,
        MAX_UPLOAD_SIZE_MB=5,
        CHUNK_SIZE=300,
        CHUNK_OVERLAP=50,
        RETRIEVAL_TOP_K=3,
        LLM_TIMEOUT_SECONDS=10.0,
        LLM_MAX_RETRIES=1,
    )


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock(spec=QdrantVectorStore)
    store.health_check.return_value = True
    store.get_document_metadata.return_value = None
    store.list_documents.return_value = []
    store.delete_by_document_id.return_value = 1
    return store


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    svc = MagicMock(spec=EmbeddingService)
    svc.get_dimension.return_value = 1536
    svc.embed_query.return_value = [0.01] * 1536
    svc.embed_documents.return_value = [[0.01] * 1536]
    return svc


@pytest.fixture
def mock_llm_service() -> MagicMock:
    svc = MagicMock(spec=LLMService)
    return svc


@pytest.fixture
def client(
    test_settings: Settings,
    mock_vector_store: MagicMock,
    mock_embedding_service: MagicMock
) -> Generator[TestClient, None, None]:
    app = create_application()

    # Apply dependency overrides
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a valid minimal PDF binary."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

