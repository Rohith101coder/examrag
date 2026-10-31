"""Schemas for RAG chat query requests, structured answers, and citations."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExamAnswer(BaseModel):
    """Structured, exam-ready answer following the 5-part academic format."""
    definition: str = Field(..., description="Concise academic definition (2-3 lines)")
    working: str = Field(..., description="Step-by-step or logical explanation of how it works")
    example: str = Field(..., description="Simple or real-world example illustrating the concept")
    advantages: List[str] = Field(default_factory=list, description="List of key advantages as bullet points")
    limitations: List[str] = Field(default_factory=list, description="List of key limitations or drawbacks")


class SourceChunk(BaseModel):
    """Citation metadata for a supporting document passage."""
    document_id: str = Field(..., description="Document ID where passage originates")
    filename: str = Field(..., description="Original filename of the source document")
    page_number: int = Field(..., description="1-indexed page number of the passage")
    chunk_index: int = Field(..., description="Zero-indexed chunk sequence number")
    text: str = Field(..., description="Supporting source passage text")
    relevance_score: float = Field(
        ...,
        description="Cosine similarity retrieval score (higher = closer match, not a probability)"
    )


class QueryMetadata(BaseModel):
    """Latency, model, and token metrics for a RAG execution."""
    model: str = Field(..., description="LLM model identifier used for generation")
    retrieved_chunks: int = Field(..., description="Number of source passages retrieved")
    retrieval_latency_ms: float = Field(..., description="Time taken to embed query and search Qdrant in ms")
    generation_latency_ms: float = Field(..., description="Time taken for LLM prompt generation in ms")
    total_latency_ms: float = Field(..., description="Total end-to-end execution latency in ms")
    input_tokens: Optional[int] = Field(default=None, description="Number of prompt/input tokens if reported by provider")
    output_tokens: Optional[int] = Field(default=None, description="Number of completion tokens if reported by provider")
    confidence_score: Optional[float] = Field(default=None, description="Retrieval-derived calibrated confidence score [0.0 - 1.0]")
    abstained: bool = Field(default=False, description="Whether the pipeline abstained from generating an answer")
    retrieval_mode: Optional[str] = Field(default="vector", description="Retrieval mechanism: 'vector', 'graph', or 'hybrid'")
    graph_subgraph: Optional[Dict[str, Any]] = Field(default=None, description="Traversed knowledge graph nodes and relational path if graph/hybrid mode")


class ChatRequest(BaseModel):
    """Query request for document-grounded answer generation."""
    question: str = Field(..., min_length=2, max_length=1000, description="The student exam question to answer")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific document IDs to restrict search to"
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant passages to retrieve (1 to 10)"
    )
    answer_mode: str = Field(
        default="exam",
        description="Answer generation format mode. Defaults to 'exam'"
    )
    retrieval_mode: Literal["vector", "graph", "hybrid"] = Field(
        default="vector",
        description="Retrieval strategy: 'vector' (Qdrant), 'graph' (NetworkX), or 'hybrid' (Reciprocal Rank Fusion)"
    )


class ChatResponse(BaseModel):
    """Response containing structured exam answer, source citations, and execution telemetry."""
    answer: ExamAnswer = Field(..., description="Structured 5-part exam-ready answer")
    sources: List[SourceChunk] = Field(default_factory=list, description="Retrieved passages used to ground the answer")
    metadata: QueryMetadata = Field(..., description="Execution telemetry and performance metrics")

