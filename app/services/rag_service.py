"""RAG orchestration service coordinating retrieval, synthesis, and telemetry."""

import logging
from typing import Optional

from app.api.schemas.chat import ChatRequest, ChatResponse, QueryMetadata
from app.core.config import Settings, get_settings
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates end-to-end question answering pipeline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_service: Optional[RetrievalService] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self.llm_service = llm_service or LLMService(settings=self.settings)

    def answer_query(self, request: ChatRequest) -> ChatResponse:
        """Execute full RAG workflow: retrieve relevant passages, generate answer, attach metadata."""
        # 1. Retrieval
        sources, retrieval_latency_ms = self.retrieval_service.search_passages(
            query=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids
        )

        # 2. Answer generation
        answer, generation_latency_ms, input_tokens, output_tokens = self.llm_service.generate_exam_answer(
            question=request.question,
            context_chunks=sources
        )

        # 3. Telemetry and telemetry aggregation
        total_latency_ms = round(retrieval_latency_ms + generation_latency_ms, 2)

        metadata = QueryMetadata(
            model=self.settings.CHAT_MODEL,
            retrieved_chunks=len(sources),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata=metadata
        )

