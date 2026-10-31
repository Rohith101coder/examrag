"""FastAPI dependency injection providers and security authenticators."""

import logging
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def verify_ingestion_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings)
) -> str:
    """
    Authenticate administrative operations (upload, path ingest, deletion).
    Uses constant-time string comparison to prevent timing attacks.
    """
    if not settings.INGESTION_API_KEY:
        logger.error("INGESTION_API_KEY is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingestion authentication is not configured on server."
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.INGESTION_API_KEY):
        logger.warning("Unauthorized access attempt with invalid or missing X-API-Key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header."
        )

    return x_api_key


def verify_chat_access(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """
    Optionally authenticate chat queries based on PUBLIC_CHAT_ENABLED setting.
    """
    if settings.PUBLIC_CHAT_ENABLED:
        return x_api_key

    # When public chat is disabled, enforce API key
    return verify_ingestion_api_key(x_api_key=x_api_key, settings=settings)


def get_vector_store(settings: Settings = Depends(get_settings)) -> QdrantVectorStore:
    return QdrantVectorStore(settings=settings)


def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    return EmbeddingService(settings=settings)


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> IngestionService:
    return IngestionService(
        settings=settings,
        vector_store=vector_store,
        embedding_service=embedding_service
    )


def get_retrieval_service(
    settings: Settings = Depends(get_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> RetrievalService:
    return RetrievalService(
        settings=settings,
        vector_store=vector_store,
        embedding_service=embedding_service
    )


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(settings=settings)


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> RAGService:
    return RAGService(
        settings=settings,
        retrieval_service=retrieval_service,
        llm_service=llm_service
    )

