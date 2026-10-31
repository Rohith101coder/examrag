"""Knowledge Graph Storage and Query Layer using NetworkX.

Provides graph persistence (JSON and GraphML), seed entity discovery,
multi-hop traversal, and chunk association for ExamRAG.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from app.graph.models import Entity, Triple

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """Production NetworkX-based academic knowledge graph store."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        # In-memory entity embeddings cache for fast seed matching: {entity_id: vector}
        self._entity_embeddings: Dict[str, List[float]] = {}

    # ========================================================================
    # Graph Construction & Modification
    # ========================================================================

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node in the graph."""
        node_id = entity.id.strip().lower()
        if self.graph.has_node(node_id):
            # Merge metadata
            existing = self.graph.nodes[node_id]
            existing_pages = set(existing.get("page_numbers", []))
            existing_pages.update(entity.page_numbers)

            existing_chunks = set(existing.get("chunk_ids", []))
            existing_chunks.update(entity.chunk_ids)

            existing_aliases = set(existing.get("aliases", []))
            existing_aliases.update(entity.aliases)

            existing_docs = set(existing.get("source_documents", []))
            existing_docs.update(entity.source_documents)

            self.graph.nodes[node_id].update({
                "page_numbers": sorted(list(existing_pages)),
                "chunk_ids": list(existing_chunks),
                "aliases": list(existing_aliases),
                "source_documents": list(existing_docs),
                "description": entity.description or existing.get("description", "")
            })
        else:
            self.graph.add_node(
                node_id,
                name=entity.name,
                entity_type=entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
                description=entity.description,
                aliases=entity.aliases,
                page_numbers=sorted(entity.page_numbers),
                source_documents=entity.source_documents,
                chunk_ids=entity.chunk_ids
            )

    def add_relation(self, triple: Triple) -> None:
        """Add a directed typed relationship between two entities."""
        src = triple.source_id.strip().lower()
        tgt = triple.target_id.strip().lower()

        # Ensure end nodes exist if missing
        if not self.graph.has_node(src):
            self.graph.add_node(src, name=triple.source_id, entity_type="Concept", aliases=[], page_numbers=[], chunk_ids=[])
        if not self.graph.has_node(tgt):
            self.graph.add_node(tgt, name=triple.target_id, entity_type="Concept", aliases=[], page_numbers=[], chunk_ids=[])

        rel_val = triple.relation.value if hasattr(triple.relation, "value") else str(triple.relation)
        self.graph.add_edge(
            src,
            tgt,
            relation=rel_val,
            weight=triple.weight,
            page_number=triple.page_number,
            evidence=triple.evidence or ""
        )

    # ========================================================================
    # Seed Entity Matching & Subgraph Traversal
    # ========================================================================

    def find_seed_entities(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        embedding_service: Optional[Any] = None,
        top_k: int = 3,
        min_similarity: float = 0.50
    ) -> List[Tuple[str, float]]:
        """Find seed entities matching query via keyword/alias and embedding cosine similarity."""
        query_lower = query.lower()
        matched: Dict[str, float] = {}

        # 1. Exact & Substring Alias Matching (high priority = 1.0)
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            aliases = [a.lower() for a in data.get("aliases", [])]

            if node_id in query_lower or name in query_lower:
                matched[node_id] = 1.0
                continue

            for alias in aliases:
                if len(alias) >= 2 and alias in query_lower:
                    matched[node_id] = max(matched.get(node_id, 0.0), 0.95)
                    break

        # 2. Embedding Cosine Similarity (if vectors available)
        if query_embedding is not None and self._entity_embeddings:
            q_norm = math.sqrt(sum(x * x for x in query_embedding))
            if q_norm > 0:
                for node_id, e_vec in self._entity_embeddings.items():
                    if node_id in self.graph:
                        e_norm = math.sqrt(sum(y * y for y in e_vec))
                        if e_norm > 0:
                            sim = sum(x * y for x, y in zip(query_embedding, e_vec)) / (q_norm * e_norm)
                            if sim >= min_similarity:
                                matched[node_id] = max(matched.get(node_id, 0.0), float(sim))

        # Sort by match score descending
        sorted_seeds = sorted(matched.items(), key=lambda x: x[1], reverse=True)
        return sorted_seeds[:top_k]

    def traverse_subgraph(
        self,
        seed_entity_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 20
    ) -> Dict[str, Any]:
        """Perform multi-hop traversal from seed entities, returning paths, nodes, and pages."""
        visited_nodes: Set[str] = set()
        visited_edges: List[Dict[str, Any]] = []
        collected_pages: Set[int] = set()
        collected_chunks: Set[str] = set()
        paths: List[List[str]] = []

        current_frontier = [s.strip().lower() for s in seed_entity_ids if self.graph.has_node(s.strip().lower())]
        visited_nodes.update(current_frontier)

        for hop in range(max_hops):
            next_frontier: Set[str] = set()
            for u in current_frontier:
                # Add node metadata
                u_data = self.graph.nodes[u]
                collected_pages.update(u_data.get("page_numbers", []))
                collected_chunks.update(u_data.get("chunk_ids", []))

                # Traverse outgoing and incoming edges for complete relational context
                neighbors = list(self.graph.successors(u)) + list(self.graph.predecessors(u))
                for v in neighbors:
                    if v not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(v)
                        next_frontier.add(v)
                        paths.append([u, v])

                    # Record edge details
                    if self.graph.has_edge(u, v):
                        edge_data = self.graph.get_edge_data(u, v)
                        visited_edges.append({
                            "source": u,
                            "target": v,
                            "relation": edge_data.get("relation", "RELATED_TO"),
                            "weight": edge_data.get("weight", 1.0),
                            "page_number": edge_data.get("page_number")
                        })
                    elif self.graph.has_edge(v, u):
                        edge_data = self.graph.get_edge_data(v, u)
                        visited_edges.append({
                            "source": v,
                            "target": u,
                            "relation": edge_data.get("relation", "RELATED_TO"),
                            "weight": edge_data.get("weight", 1.0),
                            "page_number": edge_data.get("page_number")
                        })

            current_frontier = list(next_frontier)
            if not current_frontier or len(visited_nodes) >= max_nodes:
                break

        # Final pass over all visited nodes to gather pages and chunks
        for node in visited_nodes:
            n_data = self.graph.nodes[node]
            collected_pages.update(n_data.get("page_numbers", []))
            collected_chunks.update(n_data.get("chunk_ids", []))

        return {
            "seed_entities": seed_entity_ids,
            "traversed_nodes": list(visited_nodes),
            "traversed_edges": visited_edges,
            "candidate_pages": sorted(list(collected_pages)),
            "candidate_chunk_ids": list(collected_chunks),
            "num_nodes": len(visited_nodes),
            "num_edges": len(visited_edges)
        }

    # ========================================================================
    # Serialization (JSON & GraphML)
    # ========================================================================

    def save_json(self, path: Path) -> None:
        """Serialize graph to standard node-link JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.graph, edges="edges")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Persisted knowledge graph to JSON: %s (%d nodes, %d edges)", path, self.graph.number_of_nodes(), self.graph.number_of_edges())

    def load_json(self, path: Path) -> None:
        """Load graph from node-link JSON."""
        if not path.exists():
            raise FileNotFoundError(f"Graph JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data, directed=True, edges="edges")
        logger.info("Loaded knowledge graph from JSON: %s (%d nodes, %d edges)", path, self.graph.number_of_nodes(), self.graph.number_of_edges())

    def save_graphml(self, path: Path) -> None:
        """Serialize graph to standard GraphML format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert lists to comma-separated strings for GraphML compatibility
        g_copy = self.graph.copy()
        for _, d in g_copy.nodes(data=True):
            for k, v in list(d.items()):
                if isinstance(v, list):
                    d[k] = ",".join(str(item) for item in v)
        nx.write_graphml(g_copy, str(path))
        logger.info("Persisted knowledge graph to GraphML: %s", path)

    def stats(self) -> Dict[str, Any]:
        """Compute key topological metrics of the knowledge graph."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        density = nx.density(self.graph) if n_nodes > 1 else 0.0
        avg_degree = round((2.0 * n_edges) / n_nodes, 2) if n_nodes > 0 else 0.0

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": round(density, 4),
            "average_degree": avg_degree,
            "is_directed": self.graph.is_directed()
        }
