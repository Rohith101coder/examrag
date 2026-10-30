"""PDF ingestion service for text extraction, chunking, embedding, and Qdrant indexing."""

import hashlib
import io
import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple
import uuid

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

from app.api.schemas.document import IngestionSummary
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentParsingError,
    InvalidPDFError,
    OversizedFileError,
    VectorStoreError,
)
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates PDF text extraction, chunking, embedding, and idempotent vector persistence."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or QdrantVectorStore(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)

    def ingest_pdf_file(self, file_path: Path | str) -> IngestionSummary:
        """Read a PDF from the filesystem and ingest it."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise InvalidPDFError(f"File not found or not accessible: {file_path}")

        try:
            pdf_bytes = path.read_bytes()
        except Exception as e:
            raise InvalidPDFError(f"Failed to read file from path {path}: {str(e)}") from e

        return self.ingest_pdf_bytes(pdf_bytes, filename=path.name)

    def ingest_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> IngestionSummary:
        """Process, chunk, embed, and index a PDF binary."""
        start_time = time.perf_counter()

        # 1. Enforce size limit
        max_bytes = self.settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(pdf_bytes) > max_bytes:
            raise OversizedFileError(
                f"File size ({len(pdf_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {self.settings.MAX_UPLOAD_SIZE_MB} MB."
            )

        # 2. Validate PDF signature (magic bytes)
        if not pdf_bytes.startswith(b"%PDF-"):
            raise InvalidPDFError("Uploaded file is not a valid PDF document (missing %PDF- signature).")

        # 3. Calculate SHA-256 content hash and stable document ID
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        document_id = f"doc_{content_hash[:16]}"

        # 4. Extract text per page
        page_texts, total_pages = self._extract_text_by_page(pdf_bytes)

        # 5. Split into overlapping chunks preserving page numbers
        chunks = self._chunk_pages(page_texts, document_id, filename, content_hash)
        if not chunks:
            raise DocumentParsingError("No usable text chunks could be produced from the document.")

        # 6. Idempotency Check: check if already fully indexed in Qdrant
        try:
            existing = self.vector_store.get_document_metadata(document_id)
            if existing and existing.get("content_hash") == content_hash and existing.get("chunk_count") == len(chunks):
                elapsed = round(time.perf_counter() - start_time, 3)
                logger.info(
                    "Document '%s' (%s) is already indexed with %d chunks. Skipping duplicate embedding.",
                    filename,
                    document_id,
                    len(chunks)
                )
                return IngestionSummary(
                    document_id=document_id,
                    filename=filename,
                    page_count=total_pages,
                    chunk_count=len(chunks),
                    content_hash=content_hash,
                    duration_seconds=elapsed,
                    status="already_indexed"
                )
        except Exception as e:
            # If collection doesn't exist yet, proceed with normal creation
            logger.debug("Idempotency pre-check note: %s", str(e))

        # 7. Ensure Qdrant collection exists with matching dimension
        embedding_dim = self.embedding_service.get_dimension()
        self.vector_store.ensure_collection(dimension=embedding_dim)

        # 8. Generate embeddings in batches
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts_to_embed)

        if len(embeddings) != len(chunks):
            raise VectorStoreError(
                f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)})."
            )

        # 9. Build Qdrant PointStructs with deterministic UUIDs
        points: List[models.PointStruct] = []
        for chunk_data, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"examrag:{document_id}:{chunk_data['chunk_index']}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk_data
                )
            )

        # 10. Persist vectors into Qdrant
        self.vector_store.upsert_points(points)

        elapsed = round(time.perf_counter() - start_time, 3)
        logger.info(
            "Successfully ingested document '%s' (ID: %s, pages: %d, chunks: %d) in %.2fs",
            filename,
            document_id,
            total_pages,
            len(chunks),
            elapsed
        )

        return IngestionSummary(
            document_id=document_id,
            filename=filename,
            page_count=total_pages,
            chunk_count=len(chunks),
            content_hash=content_hash,
            duration_seconds=elapsed,
            status="indexed"
        )

    def _extract_text_by_page(self, pdf_bytes: bytes) -> Tuple[List[Tuple[int, str]], int]:
        """Extract text from PDF preserving 1-indexed page numbers."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF structure: {str(e)}") from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise DocumentParsingError("PDF file contains 0 pages.")

        page_texts: List[Tuple[int, str]] = []
        total_extracted_chars = 0

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            try:
                text = page.extract_text() or ""
                # Normalize line breaks and remove null chars
                text = text.replace("\x00", "").strip()
                if text:
                    total_extracted_chars += len(text)
                    page_texts.append((page_num, text))
            except Exception as e:
                logger.warning("Error extracting text from page %d: %s", page_num, str(e))

        if total_extracted_chars == 0 or not page_texts:
            raise DocumentParsingError(
                "PDF contains no extractable text or is unreadable (scanned images without OCR are unsupported)."
            )

        return page_texts, total_pages

    def _chunk_pages(
        self,
        page_texts: List[Tuple[int, str]],
        document_id: str,
        filename: str,
        content_hash: str
    ) -> List[dict]:
        """Split page texts into overlapping chunks with metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks: List[dict] = []
        global_index = 0

        for page_num, page_content in page_texts:
            page_chunks = splitter.split_text(page_content)
            for chunk_str in page_chunks:
                clean_chunk = chunk_str.strip()
                if not clean_chunk:
                    continue

                chunks.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": global_index,
                    "text": clean_chunk,
                    "content_hash": content_hash
                })
                global_index += 1

        return chunks

