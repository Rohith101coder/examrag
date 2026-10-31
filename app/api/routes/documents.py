"""Document upload, ingestion, listing, and deletion endpoints."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.api.dependencies import get_ingestion_service, get_vector_store, verify_ingestion_api_key
from app.api.schemas.common import DeletionResponse
from app.api.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    IngestionSummary,
    IngestPathRequest,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentParsingError,
    InvalidPDFError,
    OversizedFileError,
)
from app.infrastructure.vector_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=IngestionSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
    description="Accepts a PDF file via multipart/form-data, splits it into chunks, embeds, and persists to Qdrant Cloud. Requires X-API-Key."
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    api_key: str = Depends(verify_ingestion_api_key),
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings)
) -> IngestionSummary:
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .pdf extension."
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file data: {str(e)}"
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    try:
        return service.ingest_pdf_bytes(pdf_bytes=pdf_bytes, filename=filename)
    except OversizedFileError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=e.message)
    except (InvalidPDFError, DocumentParsingError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


@router.post(
    "/ingest",
    response_model=IngestionSummary,
    status_code=status.HTTP_200_OK,
    summary="Ingest a local server-accessible PDF path",
    description="Ingests a local PDF by filesystem path. Disabled in production by default. Requires X-API-Key."
)
def ingest_local_path(
    payload: IngestPathRequest,
    api_key: str = Depends(verify_ingestion_api_key),
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings)
) -> IngestionSummary:
    if settings.APP_ENV.lower() == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Filesystem path ingestion is disabled in production environments for security."
        )

    try:
        return service.ingest_pdf_file(payload.file_path)
    except OversizedFileError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=e.message)
    except (InvalidPDFError, DocumentParsingError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    description="Retrieves a list of unique documents indexed in Qdrant Cloud."
)
def list_documents(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum documents to return"),
    vector_store: QdrantVectorStore = Depends(get_vector_store)
) -> DocumentListResponse:
    docs = vector_store.list_documents(limit=limit)
    items = [
        DocumentDetail(
            document_id=d["document_id"],
            filename=d["filename"],
            content_hash=d["content_hash"],
            chunk_count=d["chunk_count"]
        )
        for d in docs
    ]
    return DocumentListResponse(total=len(items), documents=items)


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="Get document details",
    description="Retrieve metadata and chunk count for a specific document ID."
)
def get_document(
    document_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
) -> DocumentDetail:
    meta = vector_store.get_document_metadata(document_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )

    return DocumentDetail(
        document_id=meta["document_id"],
        filename=meta["filename"],
        content_hash=meta["content_hash"],
        chunk_count=meta["chunk_count"]
    )


@router.delete(
    "/{document_id}",
    response_model=DeletionResponse,
    summary="Delete indexed document",
    description="Removes all vector points associated with the specified document ID. Requires X-API-Key."
)
def delete_document(
    document_id: str,
    api_key: str = Depends(verify_ingestion_api_key),
    vector_store: QdrantVectorStore = Depends(get_vector_store)
) -> DeletionResponse:
    deleted_points = vector_store.delete_by_document_id(document_id)
    if deleted_points == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found or already deleted."
        )

    return DeletionResponse(
        document_id=document_id,
        deleted=True,
        points_deleted=deleted_points,
        message=f"Successfully removed {deleted_points} chunks for document '{document_id}'."
    )

