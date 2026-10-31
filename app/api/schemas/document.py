"""Schemas for document management, ingestion, and listings."""

from typing import List

from pydantic import BaseModel, Field


class IngestionSummary(BaseModel):
    """Result summary of a completed PDF document ingestion."""
    document_id: str = Field(..., description="Deterministic unique document ID")
    filename: str = Field(..., description="Original filename of the ingested PDF")
    page_count: int = Field(..., description="Total pages processed")
    chunk_count: int = Field(..., description="Total chunks indexed into Qdrant")
    content_hash: str = Field(..., description="SHA-256 hash of the PDF binary")
    duration_seconds: float = Field(..., description="Processing time in seconds")
    status: str = Field(default="indexed", description="'indexed' or 'already_indexed'")


class IngestPathRequest(BaseModel):
    """Request payload for ingesting a trusted local filesystem PDF path."""
    file_path: str = Field(..., min_length=1, description="Server-accessible path to a PDF file")


class DocumentDetail(BaseModel):
    """Metadata summary of an indexed document."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original document filename")
    content_hash: str = Field(..., description="SHA-256 hash of document content")
    chunk_count: int = Field(..., description="Number of indexed chunks")


class DocumentListResponse(BaseModel):
    """Paginated or bounded list of indexed documents."""
    total: int = Field(..., description="Total number of unique documents found")
    documents: List[DocumentDetail] = Field(..., description="List of document metadata items")

