"""RAG orchestration service coordinating retrieval, synthesis, and telemetry."""

import logging
from typing import Any, Dict, Optional

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
        llm_service: Optional[LLMService] = None,
        graph_retriever: Optional[Any] = None
    ):
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self.llm_service = llm_service or LLMService(settings=self.settings)
        self._graph_retriever = graph_retriever

    def _get_graph_retriever(self):
        if self._graph_retriever is None:
            from app.graph.graph_retriever import GraphRetriever
            from eval.graph_eval import build_academic_knowledge_graph
            graph_store = build_academic_knowledge_graph(self.settings)
            self._graph_retriever = GraphRetriever(
                settings=self.settings,
                vector_store=self.retrieval_service.vector_store,
                embedding_service=self.retrieval_service.embedding_service,
                graph_store=graph_store
            )
        return self._graph_retriever

    def answer_query(self, request: ChatRequest) -> ChatResponse:
        """Execute full RAG workflow: retrieve relevant passages, generate answer, attach metadata."""
        mode = getattr(request, "retrieval_mode", "vector")
        graph_subgraph: Optional[Dict[str, Any]] = None

        # 1. Retrieval based on mode
        if mode in ("graph", "hybrid"):
            retriever = self._get_graph_retriever()
            sources, graph_meta, retrieval_latency_ms = retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                mode=mode
            )
            graph_subgraph = graph_meta
        else:
            sources, retrieval_latency_ms = self.retrieval_service.search_passages(
                query=request.question,
                top_k=request.top_k,
                document_ids=request.document_ids
            )

        # 2. Confidence calculation & Abstention Check
        top_score = sources[0].relevance_score if sources else 0.0
        # Linear calibration mapping score [0.45, 0.85] -> confidence [0.0, 1.0]
        confidence_score = round(max(0.0, min(1.0, (top_score - 0.45) / 0.40)), 4) if sources else 0.0

        should_abstain = (not sources) or (top_score < self.settings.ABSTAIN_THRESHOLD)

        if should_abstain:
            logger.info(
                "Abstaining on query (sources=%d, top_score=%.4f < threshold=%.4f)",
                len(sources),
                top_score,
                self.settings.ABSTAIN_THRESHOLD
            )
            context_passages = []
        else:
            context_passages = sources

        # 3. Answer generation
        answer, generation_latency_ms, input_tokens, output_tokens = self.llm_service.generate_exam_answer(
            question=request.question,
            context_chunks=context_passages
        )

        # 4. Telemetry and telemetry aggregation
        total_latency_ms = round(retrieval_latency_ms + generation_latency_ms, 2)
        abstained = should_abstain or ("insufficient information" in answer.definition.lower())

        metadata = QueryMetadata(
            model=self.settings.CHAT_MODEL,
            retrieved_chunks=len(sources),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            confidence_score=confidence_score,
            abstained=abstained,
            retrieval_mode=mode,
            graph_subgraph=graph_subgraph
        )

        return ChatResponse(
            answer=answer,
            sources=sources if not should_abstain else [],
            metadata=metadata
        )

