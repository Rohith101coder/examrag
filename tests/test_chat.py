"""Tests for chat querying, RAG synthesis, citations, and error resilience."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.schemas.chat import ChatResponse, ExamAnswer, QueryMetadata, SourceChunk
from app.core.config import Settings
from app.core.exceptions import (
    ProviderError,
    ProviderTimeoutError,
    StructuredOutputParsingError,
)
from app.services.rag_service import RAGService


def test_chat_query_success_with_sources(client: TestClient) -> None:
    """Successful chat query returns structured answer, citations, and metadata."""
    payload = {
        "question": "What is process scheduling in an operating system?",
        "document_ids": ["doc_os1"],
        "top_k": 3,
        "answer_mode": "exam"
    }

    mock_response = ChatResponse(
        answer=ExamAnswer(
            definition="Process scheduling is the OS mechanism of allocating CPU time to ready processes.",
            working="The scheduler selects processes from ready queue according to scheduling algorithm.",
            example="Round Robin uses a time quantum to cycle through processes.",
            advantages=["Improves throughput", "Fair CPU allocation"],
            limitations=["Context switch overhead", "Complexity"]
        ),
        sources=[
            SourceChunk(
                document_id="doc_os1",
                filename="OperatingSystems.pdf",
                page_number=4,
                chunk_index=12,
                text="Process scheduling passage content.",
                relevance_score=0.89
            )
        ],
        metadata=QueryMetadata(
            model="openai/gpt-4o-mini",
            retrieved_chunks=1,
            retrieval_latency_ms=35.2,
            generation_latency_ms=210.4,
            total_latency_ms=245.6,
            input_tokens=150,
            output_tokens=75
        )
    )

    with patch.object(RAGService, "answer_query", return_value=mock_response):
        response = client.post("/api/v1/chat/query", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Validate answer structure
        assert "definition" in data["answer"]
        assert "working" in data["answer"]
        assert "example" in data["answer"]
        assert len(data["answer"]["advantages"]) == 2
        assert len(data["answer"]["limitations"]) == 2

        # Validate citations (Scenario 12)
        assert len(data["sources"]) == 1
        src = data["sources"][0]
        assert src["document_id"] == "doc_os1"
        assert src["filename"] == "OperatingSystems.pdf"
        assert src["page_number"] == 4
        assert src["chunk_index"] == 12
        assert src["relevance_score"] == 0.89

        # Validate telemetry
        assert data["metadata"]["retrieved_chunks"] == 1
        assert data["metadata"]["total_latency_ms"] == 245.6


def test_chat_unsupported_question_abstention(client: TestClient) -> None:
    """Unsupported question returns structured abstention without fabricated facts (Scenario 11)."""
    payload = {
        "question": "Explain Roman aqueducts in 100 BCE."
    }

    mock_response = ChatResponse(
        answer=ExamAnswer(
            definition="The indexed material does not contain sufficient information to answer this question.",
            working="The retrieved document passages lack the specific details required to explain this topic.",
            example="None available from indexed context.",
            advantages=[],
            limitations=[]
        ),
        sources=[],
        metadata=QueryMetadata(
            model="openai/gpt-4o-mini",
            retrieved_chunks=0,
            retrieval_latency_ms=20.0,
            generation_latency_ms=0.0,
            total_latency_ms=20.0,
            input_tokens=None,
            output_tokens=None
        )
    )

    with patch.object(RAGService, "answer_query", return_value=mock_response):
        response = client.post("/api/v1/chat/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "does not contain sufficient information" in data["answer"]["definition"]
        assert data["sources"] == []


def test_chat_malformed_llm_output(client: TestClient) -> None:
    """Malformed LLM output triggers 502 Bad Gateway without returning fake answers (Scenario 13)."""
    payload = {"question": "Explain paging in memory."}

    with patch.object(RAGService, "answer_query", side_effect=StructuredOutputParsingError("Malformed JSON")):
        response = client.post("/api/v1/chat/query", json=payload)
        assert response.status_code == 502
        assert "unparseable response structure" in response.json()["detail"]


def test_chat_provider_timeout(client: TestClient) -> None:
    """Provider timeout triggers 504 Gateway Timeout (Scenario 14)."""
    payload = {"question": "Explain deadlock detection."}

    with patch.object(RAGService, "answer_query", side_effect=ProviderTimeoutError("OpenRouter timeout")):
        response = client.post("/api/v1/chat/query", json=payload)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]


def test_chat_provider_error(client: TestClient) -> None:
    """General provider error triggers 502 Bad Gateway (Scenario 14)."""
    payload = {"question": "Explain semaphore operations."}

    with patch.object(RAGService, "answer_query", side_effect=ProviderError("Rate limit exceeded")):
        response = client.post("/api/v1/chat/query", json=payload)
        assert response.status_code == 502
        assert "communicating with the AI model provider" in response.json()["detail"]


def test_chat_request_validation(client: TestClient) -> None:
    """Invalid requests (too short question, invalid top_k) trigger 422 (Scenario 17)."""
    # Question too short (< 2 chars)
    res1 = client.post("/api/v1/chat/query", json={"question": "?"})
    assert res1.status_code == 422

    # top_k out of range (> 10)
    res2 = client.post("/api/v1/chat/query", json={"question": "Valid question?", "top_k": 50})
    assert res2.status_code == 422


def test_private_chat_mode_enforces_auth(client: TestClient, test_settings: Settings) -> None:
    """When PUBLIC_CHAT_ENABLED=False, chat queries require valid X-API-Key (Scenario 16)."""
    test_settings.PUBLIC_CHAT_ENABLED = False

    payload = {"question": "Explain process states."}

    # Without header -> 401
    res_unauth = client.post("/api/v1/chat/query", json=payload)
    assert res_unauth.status_code == 401

    # With invalid header -> 401
    res_bad = client.post("/api/v1/chat/query", json=payload, headers={"X-API-Key": "wrong"})
    assert res_bad.status_code == 401

