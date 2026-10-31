"""Knowledge Graph Entity and Relation Extractor for ExamRAG.

Extracts structured academic entities and relational triples from document chunks
using OpenRouter LLM, followed by canonical normalization and embedding-based deduplication.
"""

import json
import logging
import math
import re
import time
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.graph.models import Entity, EntityType, GraphExtractionOutput, RelationType, Triple
from app.infrastructure.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Canonical alias mapping for standard Computer Science syllabus terms
CANONICAL_ALIAS_MAP: Dict[str, Tuple[str, EntityType]] = {
    "cpu scheduling": ("CPU Scheduling", EntityType.TOPIC),
    "processor scheduling": ("CPU Scheduling", EntityType.TOPIC),
    "tlb": ("Translation Lookaside Buffer", EntityType.CONCEPT),
    "translation lookaside buffer": ("Translation Lookaside Buffer", EntityType.CONCEPT),
    "mmu": ("Memory Management Unit", EntityType.CONCEPT),
    "memory management unit": ("Memory Management Unit", EntityType.CONCEPT),
    "fcfs": ("First-Come First-Served", EntityType.CONCEPT),
    "first-come first-served": ("First-Come First-Served", EntityType.CONCEPT),
    "sjf": ("Shortest Job First", EntityType.CONCEPT),
    "shortest job first": ("Shortest Job First", EntityType.CONCEPT),
    "rr": ("Round Robin", EntityType.CONCEPT),
    "round robin": ("Round Robin", EntityType.CONCEPT),
    "round-robin": ("Round Robin", EntityType.CONCEPT),
    "srtf": ("Shortest Remaining Time First", EntityType.CONCEPT),
    "shortest remaining time first": ("Shortest Remaining Time First", EntityType.CONCEPT),
    "deadlock": ("Deadlock", EntityType.TOPIC),
    "deadlocks": ("Deadlock", EntityType.TOPIC),
    "semaphore": ("Semaphore", EntityType.CONCEPT),
    "semaphores": ("Semaphore", EntityType.CONCEPT),
    "mutex": ("Mutual Exclusion", EntityType.CONCEPT),
    "mutual exclusion": ("Mutual Exclusion", EntityType.CONCEPT),
    "emat": ("Effective Memory Access Time", EntityType.FORMULA),
    "effective memory access time": ("Effective Memory Access Time", EntityType.FORMULA),
    "paging": ("Paging", EntityType.TOPIC),
    "page table": ("Page Table", EntityType.CONCEPT),
    "banker's algorithm": ("Banker's Algorithm", EntityType.CONCEPT),
    "bankers algorithm": ("Banker's Algorithm", EntityType.CONCEPT),
    "dining philosophers": ("Dining Philosophers Problem", EntityType.CONCEPT),
    "dining-philosophers": ("Dining Philosophers Problem", EntityType.CONCEPT),
    "coffman conditions": ("Coffman Conditions", EntityType.DEFINITION),
    "circular wait": ("Circular Wait", EntityType.DEFINITION),
    "hold and wait": ("Hold and Wait", EntityType.DEFINITION),
    "preemption": ("Preemption", EntityType.CONCEPT),
    "external fragmentation": ("External Fragmentation", EntityType.DEFINITION),
    "internal fragmentation": ("Internal Fragmentation", EntityType.DEFINITION),
    "thrashing": ("Thrashing", EntityType.CONCEPT),
    "operating systems": ("Operating Systems", EntityType.CHAPTER),
    "memory management": ("Memory Management", EntityType.CHAPTER),
    "process scheduling": ("Process Scheduling", EntityType.CHAPTER),
    "concurrency and deadlocks": ("Concurrency and Deadlocks", EntityType.CHAPTER),
}


class RawEntity(BaseModel):
    name: str = Field(..., description="Canonical entity name")
    entity_type: EntityType = Field(..., description="Concept, Topic, Formula, Definition, Chapter")
    description: str = Field(default="", description="1-sentence academic definition")
    aliases: List[str] = Field(default_factory=list, description="Synonyms, acronyms, or variants")


class RawTriple(BaseModel):
    source: str = Field(..., description="Source entity name")
    relation: RelationType = Field(..., description="DEFINED_IN, PART_OF, RELATED_TO, PREREQUISITE_OF, APPEARS_ON_PAGE")
    target: str = Field(..., description="Target entity name")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Textual snippet establishing this relationship")


class RawExtractionPayload(BaseModel):
    entities: List[RawEntity] = Field(default_factory=list)
    triples: List[RawTriple] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are an academic Computer Science Knowledge Engineer.
Extract entities and relationships from the provided textbook passage to build an academic knowledge graph.

Permitted Entity Types:
- Concept (e.g. Round Robin, TLB, Semaphore, Banker's Algorithm)
- Topic (e.g. CPU Scheduling, Paging, Deadlock Avoidance)
- Formula (e.g. Effective Memory Access Time, Turnaround Time = Completion - Arrival)
- Definition (e.g. External Fragmentation, Coffman Conditions)
- Chapter (e.g. Memory Management, Concurrency and Synchronization)

Permitted Relation Types:
- DEFINED_IN (Entity -> Chapter/Topic)
- PART_OF (Sub-concept -> Super-concept)
- RELATED_TO (Concept -> Associated Concept)
- PREREQUISITE_OF (Foundational Concept -> Advanced Concept)
- APPEARS_ON_PAGE (Entity -> Page)

Guidelines:
1. Normalize names to canonical academic terms (e.g., 'Round Robin' rather than 'the round-robin algorithm').
2. Identify multi-hop connections (e.g. 'Time Quantum' PREREQUISITE_OF 'Round Robin', 'Round Robin' RELATED_TO 'Convoy Effect').
3. Output ONLY a valid JSON object matching this schema:
{{
  "entities": [
    {{"name": "...", "entity_type": "Concept|Topic|Formula|Definition|Chapter", "description": "...", "aliases": ["..."]}}
  ],
  "triples": [
    {{"source": "...", "relation": "DEFINED_IN|PART_OF|RELATED_TO|PREREQUISITE_OF|APPEARS_ON_PAGE", "target": "...", "weight": 1.0, "evidence": "..."}}
  ]
}}
"""


class GraphExtractor:
    """Extracts, normalizes, and deduplicates entities and relations from document chunks."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        dedup_similarity_threshold: float = 0.88
    ):
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.dedup_threshold = dedup_similarity_threshold
        self._llm: Optional[ChatOpenAI] = None

    def get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.settings.OPENROUTER_API_KEY,
                base_url=self.settings.OPENROUTER_BASE_URL,
                model=self.settings.CHAT_MODEL,
                temperature=0.0,
                max_tokens=1000,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
                max_retries=2
            )
        return self._llm

    def normalize_entity_name(self, raw_name: str) -> Tuple[str, str, Optional[EntityType]]:
        """Normalize raw entity string into (canonical_name, entity_id, entity_type)."""
        cleaned = re.sub(r"[^\w\s\-\(\)]", "", raw_name).strip()
        cleaned_lower = cleaned.lower()

        if cleaned_lower in CANONICAL_ALIAS_MAP:
            canonical, etype = CANONICAL_ALIAS_MAP[cleaned_lower]
            slug = re.sub(r"\s+", "_", canonical.lower())
            return canonical, slug, etype

        # Strip leading articles
        cleaned_no_art = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", cleaned_no_art.lower()).strip("_")
        return cleaned_no_art, slug, None

    def extract_from_chunk(
        self,
        chunk_text: str,
        page_number: int,
        document_id: str,
        chunk_id: str
    ) -> GraphExtractionOutput:
        """Call LLM with structured output parsing and fallback retry."""
        llm = self.get_llm()
        user_message = f"Document: {document_id} (Page {page_number})\n\nChunk Text:\n{chunk_text}"

        payload: Optional[RawExtractionPayload] = None
        for attempt in range(3):
            try:
                res = llm.invoke([
                    HumanMessage(content=EXTRACTION_SYSTEM_PROMPT + "\n\n" + user_message)
                ])
                raw_text = res.content if isinstance(res.content, str) else str(res.content)
                fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                cand = fence.group(1).strip() if fence else raw_text[raw_text.find("{"):raw_text.rfind("}") + 1]
                data = json.loads(cand)
                payload = RawExtractionPayload(**data)
                break
            except Exception as e:
                logger.warning("Graph extraction retry %d: %s", attempt + 1, str(e))
                time.sleep(1.0)

        if payload is None:
            return GraphExtractionOutput(entities=[], triples=[])

        # Process and normalize entities
        processed_entities: List[Entity] = []
        name_to_id: Dict[str, str] = {}

        for re_ent in payload.entities:
            c_name, slug, forced_type = self.normalize_entity_name(re_ent.name)
            etype = forced_type if forced_type is not None else re_ent.entity_type
            ent_id = slug

            aliases = list(set([a.strip() for a in re_ent.aliases if a.strip()] + [re_ent.name]))
            if c_name in aliases:
                aliases.remove(c_name)

            entity = Entity(
                id=ent_id,
                name=c_name,
                entity_type=etype,
                description=re_ent.description,
                aliases=aliases,
                page_numbers=[page_number],
                source_documents=[document_id],
                chunk_ids=[chunk_id]
            )
            processed_entities.append(entity)
            name_to_id[re_ent.name.lower()] = ent_id
            name_to_id[c_name.lower()] = ent_id

        # Process and normalize triples
        processed_triples: List[Triple] = []
        for raw_tr in payload.triples:
            _, src_id, _ = self.normalize_entity_name(raw_tr.source)
            _, tgt_id, _ = self.normalize_entity_name(raw_tr.target)

            if src_id == tgt_id:
                continue

            triple = Triple(
                source_id=src_id,
                relation=raw_tr.relation,
                target_id=tgt_id,
                weight=raw_tr.weight,
                page_number=page_number,
                evidence=raw_tr.evidence
            )
            processed_triples.append(triple)

        return GraphExtractionOutput(entities=processed_entities, triples=processed_triples)

    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge identical or highly similar entities (cosine similarity > dedup_threshold)."""
        merged_by_id: Dict[str, Entity] = {}

        for ent in entities:
            if ent.id in merged_by_id:
                existing = merged_by_id[ent.id]
                existing.page_numbers = sorted(list(set(existing.page_numbers + ent.page_numbers)))
                existing.source_documents = list(set(existing.source_documents + ent.source_documents))
                existing.chunk_ids = list(set(existing.chunk_ids + ent.chunk_ids))
                existing.aliases = list(set(existing.aliases + ent.aliases))
                if not existing.description and ent.description:
                    existing.description = ent.description
            else:
                merged_by_id[ent.id] = ent

        unique_entities = list(merged_by_id.values())

        # If embedding service is available and there are multiple entities, compute embeddings
        if len(unique_entities) > 1 and self.embedding_service:
            try:
                names = [e.name for e in unique_entities]
                vectors = self.embedding_service.embed_documents(names)

                # Union-Find for clustering highly similar entities
                parent = list(range(len(unique_entities)))

                def find(i: int) -> int:
                    if parent[i] == i:
                        return i
                    parent[i] = find(parent[i])
                    return parent[i]

                def union(i: int, j: int):
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j:
                        parent[root_j] = root_i

                for i in range(len(unique_entities)):
                    for j in range(i + 1, len(unique_entities)):
                        if unique_entities[i].entity_type == unique_entities[j].entity_type:
                            v1, v2 = vectors[i], vectors[j]
                            norm1 = math.sqrt(sum(x * x for x in v1))
                            norm2 = math.sqrt(sum(y * y for y in v2))
                            if norm1 > 0 and norm2 > 0:
                                sim = sum(x * y for x, y in zip(v1, v2)) / (norm1 * norm2)
                                if sim >= self.dedup_threshold:
                                    union(i, j)

                clusters: Dict[int, List[Entity]] = {}
                for idx, ent in enumerate(unique_entities):
                    root = find(idx)
                    clusters.setdefault(root, []).append(ent)

                final_entities: List[Entity] = []
                for _, group in clusters.items():
                    canonical = group[0]
                    for other in group[1:]:
                        canonical.page_numbers = sorted(list(set(canonical.page_numbers + other.page_numbers)))
                        canonical.chunk_ids = list(set(canonical.chunk_ids + other.chunk_ids))
                        canonical.aliases = list(set(canonical.aliases + [other.name] + other.aliases))
                    final_entities.append(canonical)
                return final_entities
            except Exception as e:
                logger.warning("Embedding-based entity deduplication bypassed due to: %s", str(e))

        return unique_entities
