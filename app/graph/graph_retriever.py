"""Graph-Augmented and Hybrid Retrieval Service for ExamRAG.

Combines Qdrant dense vector search with NetworkX multi-hop knowledge graph traversal
via Reciprocal Rank Fusion (RRF). Supports 'vector', 'graph', and 'hybrid' modes.
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.api.schemas.chat import SourceChunk
from app.core.config import Settings, get_settings
from app.graph.store import KnowledgeGraphStore
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "graph", "hybrid"]


class GraphRetriever:
    """Multi-modal retriever bridging dense vector search and symbolic knowledge graph traversal."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        graph_store: Optional[KnowledgeGraphStore] = None,
    ):
        self.settings = settings or get_settings()
        self.vector_store = vector_store or QdrantVectorStore(self.settings)
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.graph_store = graph_store or KnowledgeGraphStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
        max_hops: int = 2,
        collection_name: Optional[str] = None,
        query_vector: Optional[List[float]] = None
    ) -> Tuple[List[SourceChunk], Dict[str, Any], float]:
        """Execute retrieval in vector, graph, or hybrid mode.

        Returns:
            Tuple of (retrieved_sources, graph_metadata, latency_ms)
        """
        start_time = time.perf_counter()
        active_collection = collection_name or self.settings.QDRANT_COLLECTION_NAME

        # 1. Generate query embedding if not provided
        if query_vector is None:
            query_vector = self.embedding_service.embed_query(query)

        # 2. Execute retrieval based on selected mode
        if mode == "vector":
            sources = self._retrieve_vector_only(
                query_vector=query_vector,
                top_k=top_k,
                collection_name=active_collection
            )
            graph_meta = {"mode": "vector", "seeds": [], "traversed_nodes": [], "edges": []}

        elif mode == "graph":
            sources, graph_meta = self._retrieve_graph_only(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                max_hops=max_hops,
                collection_name=active_collection
            )
            graph_meta["mode"] = "graph"

        else:  # Hybrid mode (Vector + Graph RRF)
            sources, graph_meta = self._retrieve_hybrid(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                max_hops=max_hops,
                collection_name=active_collection
            )
            graph_meta["mode"] = "hybrid"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return sources, graph_meta, latency_ms

    def _retrieve_vector_only(
        self,
        query_vector: List[float],
        top_k: int,
        collection_name: str
    ) -> List[SourceChunk]:
        """Perform standard dense vector similarity search via Qdrant."""
        scored_points = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=self.settings.RELEVANCE_THRESHOLD,
            collection_name=collection_name
        )
        return [
            SourceChunk(
                document_id=point.payload.get("document_id", "unknown"),
                filename=point.payload.get("filename", "unknown.pdf"),
                page_number=int(point.payload.get("page_number", 1)),
                chunk_index=int(point.payload.get("chunk_index", 0)),
                relevance_score=round(float(point.score), 4),
                text=point.payload.get("text", "")
            )
            for point in scored_points
        ]

    def _retrieve_graph_only(
        self,
        query: str,
        query_vector: List[float],
        top_k: int,
        max_hops: int,
        collection_name: str
    ) -> Tuple[List[SourceChunk], Dict[str, Any]]:
        """Perform graph-guided seed matching and multi-hop neighborhood retrieval."""
        seeds = self.graph_store.find_seed_entities(
            query=query,
            query_embedding=query_vector,
            top_k=3,
            min_similarity=0.45
        )
        seed_ids = [s[0] for s in seeds]

        if not seed_ids:
            # Fallback to vector search if no graph entities match
            sources = self._retrieve_vector_only(query_vector, top_k, collection_name)
            return sources, {"fallback": "no_graph_seeds_found", "seeds": []}

        subgraph = self.graph_store.traverse_subgraph(
            seed_entity_ids=seed_ids,
            max_hops=max_hops,
            max_nodes=15
        )

        candidate_pages = subgraph.get("candidate_pages", [])

        # Retrieve vectors from Qdrant with candidate page preference
        raw_vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 3,
            score_threshold=0.0,
            collection_name=collection_name
        )

        # Prioritize passages from graph-visited pages
        matched_sources: List[SourceChunk] = []
        other_sources: List[SourceChunk] = []

        for p in raw_vector_results:
            doc_page = int(p.payload.get("page_number", 1))
            doc = SourceChunk(
                document_id=p.payload.get("document_id", "unknown"),
                filename=p.payload.get("filename", "unknown.pdf"),
                page_number=doc_page,
                chunk_index=int(p.payload.get("chunk_index", 0)),
                relevance_score=round(float(p.score), 4),
                text=p.payload.get("text", "")
            )
            if doc_page in candidate_pages:
                matched_sources.append(doc)
            else:
                other_sources.append(doc)

        final_sources = (matched_sources + other_sources)[:top_k]
        return final_sources, subgraph

    def _retrieve_hybrid(
        self,
        query: str,
        query_vector: List[float],
        top_k: int,
        max_hops: int,
        collection_name: str
    ) -> Tuple[List[SourceChunk], Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) between dense vector search and graph-guided search."""
        # 1. Vector Search Candidates (rank list 1)
        vector_candidates = self._retrieve_vector_only(
            query_vector=query_vector,
            top_k=top_k * 2,
            collection_name=collection_name
        )

        # 2. Graph Traversal Candidates (rank list 2)
        graph_candidates, graph_meta = self._retrieve_graph_only(
            query=query,
            query_vector=query_vector,
            top_k=top_k * 2,
            max_hops=max_hops,
            collection_name=collection_name
        )

        # 3. Reciprocal Rank Fusion: RRF(d) = sum(1 / (k_rrf + rank))
        k_rrf = 60.0
        scores: Dict[Tuple[str, int], float] = {}
        doc_store: Dict[Tuple[str, int], SourceChunk] = {}

        for rank, doc in enumerate(vector_candidates):
            key = (doc.filename, doc.page_number)
            scores[key] = scores.get(key, 0.0) + (1.0 / (k_rrf + rank + 1))
            doc_store[key] = doc

        for rank, doc in enumerate(graph_candidates):
            key = (doc.filename, doc.page_number)
            scores[key] = scores.get(key, 0.0) + (1.0 / (k_rrf + rank + 1))
            if key not in doc_store:
                doc_store[key] = doc

        # Sort combined results by RRF score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        fused_sources = [doc_store[k] for k in sorted_keys[:top_k]]

        return fused_sources, graph_meta
