"""Retrieval service for semantic search, passage ranking, and citation extraction."""

import logging
import time
from typing import List, Optional, Tuple

from app.api.schemas.chat import SourceChunk
from app.core.config import Settings, get_settings
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Performs semantic similarity search against indexed document chunks in Qdrant."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or QdrantVectorStore(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)

    def search_passages(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> Tuple[List[SourceChunk], float]:
        """
        Embed user query, execute vector search, and return mapped source citations with latency.
        """
        start_time = time.perf_counter()
        effective_top_k = top_k or self.settings.RETRIEVAL_TOP_K
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.RELEVANCE_THRESHOLD
        )

        # 1. Embed query vector
        query_vector = self.embedding_service.embed_query(query)

        # 2. Search Qdrant vector store
        scored_points = self.vector_store.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            document_ids=document_ids,
            score_threshold=effective_threshold
        )

        # 3. Map scored points to domain SourceChunk citations
        sources: List[SourceChunk] = []
        for point in scored_points:
            payload = point.payload or {}
            sources.append(
                SourceChunk(
                    document_id=payload.get("document_id", "unknown_doc"),
                    filename=payload.get("filename", "unknown.pdf"),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    relevance_score=round(float(point.score), 4)
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.debug(
            "Retrieved %d chunks for query in %.2f ms (doc_filter=%s)",
            len(sources),
            latency_ms,
            document_ids
        )

        return sources, latency_ms

