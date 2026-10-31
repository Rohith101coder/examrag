"""Tests for internal domain services: ingestion, vector store, retrieval, and evaluation metrics."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models

from app.api.schemas.chat import ChatRequest, ExamAnswer
from app.core.config import Settings
from app.core.exceptions import (
    CollectionDimensionMismatchError,
    DocumentParsingError,
)
from app.infrastructure.vector_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from evaluation.evaluate import (
    calculate_fact_coverage,
    calculate_mrr,
    calculate_recall_at_k,
    verify_abstention,
)


def test_collection_dimension_mismatch(test_settings: Settings) -> None:
    """Vector store must raise CollectionDimensionMismatchError if existing collection dimension differs (Scenario 8)."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Simulate existing collection configured for 768 dimensions
    existing_collection_info = MagicMock()
    existing_collection_info.config.params.vectors = models.VectorParams(size=768, distance=models.Distance.COSINE)
    mock_client.get_collection.return_value = existing_collection_info

    store = QdrantVectorStore(settings=test_settings, client=mock_client)

    # Attempting to ensure collection with dimension 1536 must raise error
    with pytest.raises(CollectionDimensionMismatchError) as exc_info:
        store.ensure_collection(dimension=1536)

    assert "dimension mismatch" in str(exc_info.value).lower()
    assert "768" in str(exc_info.value)
    assert "1536" in str(exc_info.value)


def test_retrieval_service_document_filtering(test_settings: Settings) -> None:
    """Retrieval service must pass document_ids filter to vector store (Scenario 10)."""
    mock_vector_store = MagicMock()
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1] * 1536
    mock_vector_store.search.return_value = [
        models.ScoredPoint(
            id="test-id",
            version=1,
            score=0.88,
            payload={
                "document_id": "doc_target",
                "filename": "TargetOS.pdf",
                "page_number": 2,
                "chunk_index": 5,
                "text": "Relevant passage text"
            }
        )
    ]

    retrieval_service = RetrievalService(
        settings=test_settings,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service
    )

    sources, latency_ms = retrieval_service.search_passages(
        query="What is a scheduler?",
        top_k=2,
        document_ids=["doc_target"]
    )

    # Assert search was called with document_ids filter
    mock_vector_store.search.assert_called_once_with(
        query_vector=[0.1] * 1536,
        top_k=2,
        document_ids=["doc_target"],
        score_threshold=None
    )
    assert len(sources) == 1
    assert sources[0].document_id == "doc_target"
    assert sources[0].page_number == 2


def test_empty_collection_rag_abstention(test_settings: Settings) -> None:
    """When vector store returns no results, RAG pipeline must return abstention (Scenario 9)."""
    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = []

    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1] * 1536

    retrieval_service = RetrievalService(
        settings=test_settings,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service
    )
    llm_service = LLMService(settings=test_settings)
    rag_service = RAGService(
        settings=test_settings,
        retrieval_service=retrieval_service,
        llm_service=llm_service
    )

    req = ChatRequest(question="What is CPU scheduling?", top_k=3)
    resp = rag_service.answer_query(req)

    assert resp.sources == []
    assert "does not contain sufficient information" in resp.answer.definition
    assert resp.metadata.retrieved_chunks == 0


def test_ingestion_text_extraction_empty_pdf(test_settings: Settings) -> None:
    """PDF with 0 extractable characters must raise DocumentParsingError (Scenario 4)."""
    mock_store = MagicMock()
    mock_emb = MagicMock()
    service = IngestionService(settings=test_settings, vector_store=mock_store, embedding_service=mock_emb)

    with patch.object(service, "_extract_text_by_page", side_effect=DocumentParsingError("PDF contains no extractable text")):
        with pytest.raises(DocumentParsingError):
            service.ingest_pdf_bytes(b"%PDF-1.4 header without text", "empty.pdf")


def test_evaluation_metric_calculations() -> None:
    """Unit tests for evaluation metric formulas (Scenario 18)."""
    # 1. Recall@k
    retrieved = [1, 3, 5]
    relevant = [1, 2]
    # In top 3: page 1 is hit -> 1/2 = 0.5
    recall_3 = calculate_recall_at_k(retrieved, relevant, k=3)
    assert recall_3 == 0.5

    # Full hit
    assert calculate_recall_at_k([1, 2, 4], [1, 2], k=3) == 1.0

    # 2. MRR (Mean Reciprocal Rank)
    # First relevant page (2) is at rank 2 -> 1/2 = 0.5
    mrr = calculate_mrr([4, 2, 1], [1, 2])
    assert mrr == 0.5

    # First relevant page at rank 1 -> 1/1 = 1.0
    assert calculate_mrr([1, 4, 5], [1, 2]) == 1.0

    # No relevant pages found -> 0.0
    assert calculate_mrr([7, 8, 9], [1, 2]) == 0.0

    # 3. Fact coverage
    answer_prose = "Process scheduling allocates CPU time to ready processes using Round Robin with time quantum."
    expected_facts = ["Process scheduling", "time quantum", "nonexistent fact"]
    # 2 of 3 match -> 2/3 = 0.6667
    assert calculate_fact_coverage(answer_prose, expected_facts) == 0.6667

    # 4. Abstention verification
    abstain_answer = ExamAnswer(
        definition="The indexed material does not contain sufficient information to answer this question.",
        working="The retrieved document passages lack the specific details required to explain this topic.",
        example="None available from indexed context.",
        advantages=[],
        limitations=[]
    )
    assert verify_abstention(abstain_answer) is True

    answered = ExamAnswer(
        definition="Process scheduling is CPU allocation.",
        working="Step by step execution.",
        example="Round Robin.",
        advantages=["Fairness"],
        limitations=["Overhead"]
    )
    assert verify_abstention(answered) is False

