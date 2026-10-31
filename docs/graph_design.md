# Knowledge Graph Architecture & Design Specification: ExamRAG

## 1. Objective and Academic Motivation

Retrieval-Augmented Generation (RAG) applications in academic Computer Science domains struggle with multi-hop questions (e.g., *"How does the Round Robin time quantum value directly cause or mitigate the FCFS Convoy Effect?"*). Standard vector search maps queries into a continuous embedding space, but frequently retrieves isolated chunks that mention only one facet of the problem (e.g., page 2 describing Convoy Effect or page 3 describing Round Robin) without retrieving the relational bridge between them.

ExamRAG incorporates a **hybrid Knowledge Graph layer** to explicitly preserve hierarchical curriculum structure, formula dependencies, and conceptual relationships across textbook chapters.

---

## 2. Ontology & Schema Specification

The ExamRAG ontology uses a strict Pydantic-validated schema tailored for academic CS textbooks:

### 2.1 Entity Types
| Entity Type | Description | Canonical Examples |
| :--- | :--- | :--- |
| `Concept` | Core algorithmic mechanisms or OS entities | `Round Robin`, `TLB`, `Semaphore`, `Banker's Algorithm` |
| `Topic` | Broad curricular sections or problem domains | `CPU Scheduling`, `Paging`, `Deadlock Avoidance` |
| `Formula` | Mathematical relationships and expressions | `Effective Memory Access Time (EMAT)`, `Turnaround Time = Completion - Arrival` |
| `Definition` | Standard formal textbook definitions | `External Fragmentation`, `Coffman Conditions`, `Race Condition` |
| `Chapter` | High-level textbook modules | `Process Scheduling`, `Memory Management`, `Concurrency and Deadlocks` |

### 2.2 Relationship Types
| Relation Type | Directed Semantics | Academic Example |
| :--- | :--- | :--- |
| `DEFINED_IN` | Entity $\rightarrow$ Topic / Chapter | `Convoy Effect` $\xrightarrow{\text{DEFINED_IN}}$ `CPU Scheduling` |
| `PART_OF` | Sub-component $\rightarrow$ Super-component | `Translation Lookaside Buffer` $\xrightarrow{\text{PART_OF}}$ `Memory Management Unit` |
| `RELATED_TO` | Conceptual co-occurrence or tradeoff | `Round Robin` $\xrightarrow{\text{RELATED_TO}}$ `First-Come First-Served` |
| `PREREQUISITE_OF` | Foundational $\rightarrow$ Dependent Concept | `Paging` $\xrightarrow{\text{PREREQUISITE_OF}}$ `Virtual Memory` |
| `APPEARS_ON_PAGE` | Entity $\rightarrow$ Page Number | `Coffman Conditions` $\xrightarrow{\text{APPEARS_ON_PAGE}}$ `Page 2 (Deadlocks)` |

---

## 3. Entity Normalization & Deduplication Pipeline

Raw LLM extractions suffer from synonymous variation (e.g., *"the Round-Robin algorithm"*, *"RR scheduling"*, *"Round Robin"*). ExamRAG enforces a two-tier deduplication pipeline:

1. **Canonical Dictionary Normalization**: A deterministic CS dictionary maps acronyms (`TLB`, `MMU`, `FCFS`, `SJF`, `RR`, `EMAT`) to standardized textbook terminology.
2. **Embedding-Based Cosine Clustering**: For terms not in the alias map, pairwise cosine similarity between entity name embeddings is computed. If $\text{sim}(e_1, e_2) \ge 0.88$ and their `entity_type` matches, a Union-Find disjoint-set data structure merges them into a single canonical node, aggregating aliases, page numbers, and chunk references.

---

## 4. Architectural Tradeoff: NetworkX vs. Neo4j

| Dimension | NetworkX In-Memory (`ExamRAG` Choice) | Neo4j Graph Database |
| :--- | :--- | :--- |
| **Operational Overhead** | Zero external dependencies; embeds directly into the Python process. | Requires dedicated JVM service, port exposure, and cluster provisioning. |
| **Query Latency (1-2 hops)** | Sub-millisecond ($< 0.5\,\text{ms}$) via native Python adjacency dicts. | $5\text{--}25\,\text{ms}$ network roundtrip via Bolt protocol + Cypher compilation. |
| **Deployment Footprint** | Extremely lightweight; serialized to portable JSON/GraphML artifacts. | Memory-heavy (minimum 1GB RAM heap required for Neo4j JVM instance). |
| **Graph Scale Limits** | Up to $\approx 500,000$ nodes in memory; ideal for course textbooks. | Billions of nodes with ACID transactions and distributed partitioning. |
| **Query Flexibility** | Native Python functional graph traversals and custom BFS/DFS logic. | Declarative Cypher queries (`MATCH (p)-[:PREREQUISITE_OF*1..2]->(q)`). |
| **Cold Startup Time** | $< 50\,\text{ms}$ to load node-link JSON from disk. | $10\text{--}30\,\text{s}$ container initialization time. |

### Rationale
For ExamRAG's target domain—targeted university syllabus document sets and interactive Q&A—**NetworkX** provides optimal performance: deterministic execution, zero network hops, and simple deployment on stateless cloud hosts like Render without requiring an expensive managed database instance.

---

## 5. Graph-Augmented & Hybrid Retrieval Strategy

ExamRAG implements **Reciprocal Rank Fusion (RRF)** combining vector and graph retrieval:
1. **Dense Vector Search**: Identifies chunks semantically similar to the student's query via Qdrant Cloud.
2. **Graph Traversal**: 
   - Resolves seed entities using alias matching and query embedding similarity ($\ge 0.45$).
   - Traverses $1\text{--}2$ hops along `PREREQUISITE_OF`, `PART_OF`, and `RELATED_TO` edges to capture context bridging multi-page dependencies.
   - Extracts candidate page numbers and chunk IDs associated with the traversed subgraph.
3. **Fusion (RRF)**:
   $$RRF(d) = \sum_{m \in \{\text{vector}, \text{graph}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   Passages located on graph-traversed pages receive a rank promotion, directly overcoming the blind spots of vector search on multi-hop questions.
