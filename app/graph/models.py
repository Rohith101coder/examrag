"""Knowledge Graph Data Models and Schema Definitions for ExamRAG.

Defines strict Pydantic schemas for entities, relations (triples),
and extraction outputs aligned with B.Tech Computer Science curricular knowledge.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Permitted entity types in the ExamRAG academic knowledge graph."""
    CONCEPT = "Concept"
    TOPIC = "Topic"
    FORMULA = "Formula"
    DEFINITION = "Definition"
    CHAPTER = "Chapter"


class RelationType(str, Enum):
    """Strict academic relations governing conceptual and curricular dependency."""
    DEFINED_IN = "DEFINED_IN"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    APPEARS_ON_PAGE = "APPEARS_ON_PAGE"


class Entity(BaseModel):
    """An academic entity within the knowledge graph."""
    id: str = Field(..., description="Normalized unique identifier (e.g. lowercase slug)")
    name: str = Field(..., description="Canonical display name of the entity")
    entity_type: EntityType = Field(..., description="Category of the entity")
    description: str = Field(default="", description="Concise academic summary of the entity")
    aliases: List[str] = Field(default_factory=list, description="Common synonyms or abbreviations (e.g. ['RR', 'Round-Robin'])")
    page_numbers: List[int] = Field(default_factory=list, description="Document page numbers where this entity is discussed")
    source_documents: List[str] = Field(default_factory=list, description="Document filenames containing this entity")
    chunk_ids: List[str] = Field(default_factory=list, description="Originating chunk IDs for grounded vector-graph fusion")


class Triple(BaseModel):
    """A directed semantic relation between two entities."""
    source_id: str = Field(..., description="Identifier of the source entity")
    relation: RelationType = Field(..., description="Typed predicate relation")
    target_id: str = Field(..., description="Identifier of the target entity")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence or relationship strength")
    page_number: Optional[int] = Field(default=None, description="Page number providing evidence for the triple")
    evidence: Optional[str] = Field(default=None, description="Snippet or textual sentence justifying the link")


class GraphExtractionOutput(BaseModel):
    """Structured extraction output from LLM for a text chunk."""
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    triples: List[Triple] = Field(default_factory=list, description="Extracted relationships between entities")
