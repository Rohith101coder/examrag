"""Unit tests for the Knowledge Graph layer, entity extraction, traversal, and /ask endpoint.

All tests run in-memory with mocked LLM and vector store (no external network calls).
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.schemas.chat import ChatResponse, ExamAnswer, QueryMetadata, SourceChunk
from app.graph.extractor import GraphExtractor
from app.graph.models import Entity, EntityType, RelationType, Triple
from app.graph.store import KnowledgeGraphStore
from app.main import create_application


class TestEntityNormalizationAndDedup:
    """Tests for canonical dictionary mapping, slug generation, and deduplication."""

    @pytest.fixture
    def extractor(self):
        mock_settings = MagicMock()
        mock_settings.OPENROUTER_API_KEY = "mock-key"
        mock_settings.OPENROUTER_BASE_URL = "https://mock.openrouter.ai"
        mock_settings.CHAT_MODEL = "openai/gpt-4o-mini"
        mock_settings.LLM_TIMEOUT_SECONDS = 10.0
        return GraphExtractor(settings=mock_settings, embedding_service=None)

    def test_canonical_acronym_normalization(self, extractor):
        name, slug, etype = extractor.normalize_entity_name("TLB")
        assert name == "Translation Lookaside Buffer"
        assert slug == "translation_lookaside_buffer"
        assert etype == EntityType.CONCEPT

        name, slug, etype = extractor.normalize_entity_name("round robin")
        assert name == "Round Robin"
        assert slug == "round_robin"
        assert etype == EntityType.CONCEPT

        name, slug, etype = extractor.normalize_entity_name("EMAT")
        assert name == "Effective Memory Access Time"
        assert etype == EntityType.FORMULA

    def test_article_stripping_and_slugification(self, extractor):
        name, slug, etype = extractor.normalize_entity_name("The Banker's Algorithm")
        # Should match alias map or strip 'The'
        assert "banker" in slug
        assert not slug.startswith("the_")

    def test_entity_deduplication_by_id(self, extractor):
        e1 = Entity(
            id="round_robin",
            name="Round Robin",
            entity_type=EntityType.CONCEPT,
            description="Preemptive algorithm",
            aliases=["RR"],
            page_numbers=[3],
            chunk_ids=["chunk_1"]
        )
        e2 = Entity(
            id="round_robin",
            name="Round Robin",
            entity_type=EntityType.CONCEPT,
            description="Preemptive scheduling based on time quantum",
            aliases=["Round-Robin"],
            page_numbers=[4],
            chunk_ids=["chunk_2"]
        )
        merged = extractor.deduplicate_entities([e1, e2])
        assert len(merged) == 1
        assert sorted(merged[0].page_numbers) == [3, 4]
        assert set(merged[0].aliases) == {"RR", "Round-Robin"}
        assert set(merged[0].chunk_ids) == {"chunk_1", "chunk_2"}


class TestKnowledgeGraphStore:
    """NetworkX storage, multi-hop traversal, and serialization tests."""

    @pytest.fixture
    def populated_store(self) -> KnowledgeGraphStore:
        store = KnowledgeGraphStore()

        # Add nodes
        store.add_entity(Entity(
            id="time_quantum",
            name="Time Quantum",
            entity_type=EntityType.CONCEPT,
            page_numbers=[3],
            aliases=["time slice"]
        ))
        store.add_entity(Entity(
            id="round_robin",
            name="Round Robin",
            entity_type=EntityType.CONCEPT,
            page_numbers=[3],
            aliases=["RR"]
        ))
        store.add_entity(Entity(
            id="convoy_effect",
            name="Convoy Effect",
            entity_type=EntityType.DEFINITION,
            page_numbers=[2]
        ))
        store.add_entity(Entity(
            id="fcfs",
            name="First-Come First-Served",
            entity_type=EntityType.CONCEPT,
            page_numbers=[2],
            aliases=["FCFS"]
        ))

        # Add directed relations
        store.add_relation(Triple(
            source_id="time_quantum",
            relation=RelationType.PREREQUISITE_OF,
            target_id="round_robin",
            page_number=3
        ))
        store.add_relation(Triple(
            source_id="round_robin",
            relation=RelationType.RELATED_TO,
            target_id="convoy_effect",
            page_number=3
        ))
        store.add_relation(Triple(
            source_id="convoy_effect",
            relation=RelationType.DEFINED_IN,
            target_id="fcfs",
            page_number=2
        ))
        return store

    def test_seed_entity_matching(self, populated_store):
        # Exact keyword match
        seeds = populated_store.find_seed_entities("What is Round Robin?")
        seed_ids = [s[0] for s in seeds]
        assert "round_robin" in seed_ids

        # Alias match
        seeds_alias = populated_store.find_seed_entities("Explain RR scheduling")
        seed_ids_alias = [s[0] for s in seeds_alias]
        assert "round_robin" in seed_ids_alias

    def test_multi_hop_traversal(self, populated_store):
        # 1-hop traversal from 'round_robin'
        subgraph_1hop = populated_store.traverse_subgraph(["round_robin"], max_hops=1)
        assert "round_robin" in subgraph_1hop["traversed_nodes"]
        # Connected: time_quantum (predecessor) and convoy_effect (successor)
        assert "convoy_effect" in subgraph_1hop["traversed_nodes"]
        assert "time_quantum" in subgraph_1hop["traversed_nodes"]
        assert "fcfs" not in subgraph_1hop["traversed_nodes"]

        # 2-hop traversal reaches 'fcfs' through convoy_effect
        subgraph_2hop = populated_store.traverse_subgraph(["round_robin"], max_hops=2)
        assert "fcfs" in subgraph_2hop["traversed_nodes"]
        assert set(subgraph_2hop["candidate_pages"]) == {2, 3}

    def test_json_and_graphml_serialization(self, populated_store, tmp_path):
        json_path = tmp_path / "test_graph.json"
        graphml_path = tmp_path / "test_graph.graphml"

        populated_store.save_json(json_path)
        assert json_path.exists()

        new_store = KnowledgeGraphStore()
        new_store.load_json(json_path)
        assert new_store.graph.number_of_nodes() == populated_store.graph.number_of_nodes()
        assert new_store.graph.number_of_edges() == populated_store.graph.number_of_edges()

        populated_store.save_graphml(graphml_path)
        assert graphml_path.exists()


class TestAskEndpoint:
    """FastAPI integration tests for /ask?mode=vector|graph|hybrid."""

    def test_ask_endpoint_hybrid_mode(self):
        from app.api.dependencies import get_rag_service

        mock_rag = MagicMock()
        mock_rag.answer_query.return_value = ChatResponse(
            answer=ExamAnswer(
                definition="Round Robin uses preemption with a time slice.",
                working="Processes execute for at most time quantum Q.",
                example="Q = 20ms across 3 CPU bound jobs.",
                advantages=["Fair allocation", "Prevents starvation"],
                limitations=["High context switch overhead if Q is tiny"]
            ),
            sources=[
                SourceChunk(
                    document_id="doc_123",
                    filename="os_process_scheduling.pdf",
                    page_number=3,
                    chunk_index=1,
                    text="Round Robin allocates each process a small time slice.",
                    relevance_score=0.88
                )
            ],
            metadata=QueryMetadata(
                model="openai/gpt-4o-mini",
                retrieved_chunks=1,
                retrieval_latency_ms=15.0,
                generation_latency_ms=120.0,
                total_latency_ms=135.0,
                confidence_score=0.92,
                abstained=False,
                retrieval_mode="hybrid"
            )
        )

        app = create_application()
        app.dependency_overrides[get_rag_service] = lambda: mock_rag
        client = TestClient(app)

        response = client.get("/ask?query=How+does+Round+Robin+work&mode=hybrid")
        assert response.status_code == 200
        data = response.json()
        assert "Round Robin" in data["answer"]["definition"]
        assert data["metadata"]["retrieval_mode"] == "hybrid"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["page_number"] == 3

    def test_ask_endpoint_vector_mode(self):
        from app.api.dependencies import get_rag_service

        mock_rag = MagicMock()
        mock_rag.answer_query.return_value = ChatResponse(
            answer=ExamAnswer(
                definition="Vector answer definition.",
                working="Working explanation.",
                example="Example.",
                advantages=["Adv 1"],
                limitations=["Lim 1"]
            ),
            sources=[],
            metadata=QueryMetadata(
                model="openai/gpt-4o-mini",
                retrieved_chunks=0,
                retrieval_latency_ms=10.0,
                generation_latency_ms=80.0,
                total_latency_ms=90.0,
                confidence_score=0.5,
                abstained=False,
                retrieval_mode="vector"
            )
        )

        app = create_application()
        app.dependency_overrides[get_rag_service] = lambda: mock_rag
        client = TestClient(app)

        response = client.get("/ask?query=Test+query&mode=vector")
        assert response.status_code == 200
        assert response.json()["metadata"]["retrieval_mode"] == "vector"
