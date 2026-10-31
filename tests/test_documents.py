"""Tests for document upload, ingestion, listing, and deletion endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.schemas.document import IngestionSummary
from app.core.config import Settings
from app.services.ingestion_service import IngestionService


def test_upload_missing_api_key(client: TestClient) -> None:
    """Upload endpoint must require X-API-Key."""
    files = {"file": ("test.pdf", b"%PDF-1.4 mock content", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 401
    assert "Invalid or missing X-API-Key" in response.json()["detail"]


def test_upload_invalid_api_key(client: TestClient) -> None:
    """Upload endpoint must reject invalid X-API-Key."""
    headers = {"X-API-Key": "wrong-secret-key"}
    files = {"file": ("test.pdf", b"%PDF-1.4 mock content", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 401


def test_upload_non_pdf_extension(client: TestClient) -> None:
    """Files without .pdf extension must be rejected with 400."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    files = {"file": ("notes.txt", b"plain text notes", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "must have a .pdf extension" in response.json()["detail"]


def test_upload_empty_pdf(client: TestClient) -> None:
    """0-byte file must be rejected with 400."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_invalid_pdf_magic_bytes(client: TestClient) -> None:
    """Files without %PDF- signature must be rejected with 400."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    files = {"file": ("corrupt.pdf", b"NOT_A_PDF_HEADER", "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_oversized_pdf(client: TestClient, test_settings: Settings) -> None:
    """Uploads exceeding MAX_UPLOAD_SIZE_MB must be rejected with 413."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    # MAX_UPLOAD_SIZE_MB is 5 in test_settings; create 6MB buffer
    large_payload = b"%PDF-1.4" + (b"0" * (6 * 1024 * 1024))
    files = {"file": ("large.pdf", large_payload, "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files, headers=headers)
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_upload_successful_ingestion(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Valid PDF upload must invoke ingestion service and return 201 with summary."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    files = {"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")}

    mock_summary = IngestionSummary(
        document_id="doc_mock123",
        filename="sample.pdf",
        page_count=3,
        chunk_count=12,
        content_hash="abc123hash",
        duration_seconds=0.45,
        status="indexed"
    )

    with patch.object(IngestionService, "ingest_pdf_bytes", return_value=mock_summary):
        response = client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc_mock123"
        assert data["chunk_count"] == 12
        assert data["status"] == "indexed"


def test_duplicate_ingestion_behavior(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Repeated ingestion of identical PDF returns already_indexed status idempotently."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    files = {"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")}

    mock_summary = IngestionSummary(
        document_id="doc_mock123",
        filename="sample.pdf",
        page_count=3,
        chunk_count=12,
        content_hash="abc123hash",
        duration_seconds=0.05,
        status="already_indexed"
    )

    with patch.object(IngestionService, "ingest_pdf_bytes", return_value=mock_summary):
        response = client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "already_indexed"


def test_ingest_path_disabled_in_production(client: TestClient, test_settings: Settings) -> None:
    """Local filesystem path ingestion must return 403 in production environment."""
    test_settings.APP_ENV = "production"
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    payload = {"file_path": "/safe/local/path.pdf"}

    response = client.post("/api/v1/documents/ingest", json=payload, headers=headers)
    assert response.status_code == 403
    assert "disabled in production" in response.json()["detail"]


def test_list_documents(client: TestClient, mock_vector_store: MagicMock) -> None:
    """GET /api/v1/documents lists unique documents aggregated from Qdrant."""
    mock_vector_store.list_documents.return_value = [
        {
            "document_id": "doc_os1",
            "filename": "OperatingSystems.pdf",
            "content_hash": "hash1",
            "chunk_count": 25
        }
    ]

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["documents"][0]["document_id"] == "doc_os1"
    assert data["documents"][0]["chunk_count"] == 25


def test_get_document_detail_success(client: TestClient, mock_vector_store: MagicMock) -> None:
    """GET /api/v1/documents/{document_id} returns document metadata."""
    mock_vector_store.get_document_metadata.return_value = {
        "document_id": "doc_os1",
        "filename": "OperatingSystems.pdf",
        "content_hash": "hash1",
        "chunk_count": 25
    }

    response = client.get("/api/v1/documents/doc_os1")
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc_os1"


def test_get_document_detail_not_found(client: TestClient, mock_vector_store: MagicMock) -> None:
    """GET /api/v1/documents/{document_id} returns 404 when document does not exist."""
    mock_vector_store.get_document_metadata.return_value = None

    response = client.get("/api/v1/documents/doc_nonexistent")
    assert response.status_code == 404


def test_delete_document_success(client: TestClient, mock_vector_store: MagicMock) -> None:
    """DELETE /api/v1/documents/{document_id} removes points and requires X-API-Key."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    mock_vector_store.delete_by_document_id.return_value = 15

    response = client.delete("/api/v1/documents/doc_os1", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True
    assert data["points_deleted"] == 15


def test_delete_document_not_found(client: TestClient, mock_vector_store: MagicMock) -> None:
    """DELETE /api/v1/documents/{document_id} returns 404 if document has 0 points."""
    headers = {"X-API-Key": "test-secret-ingestion-key"}
    mock_vector_store.delete_by_document_id.return_value = 0

    response = client.delete("/api/v1/documents/doc_empty", headers=headers)
    assert response.status_code == 404

