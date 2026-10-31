# Golden Evaluation Dataset (`eval/golden/`)

This directory contains the standardized evaluation benchmark for **ExamRAG**, designed to evaluate retrieval precision/recall, generation faithfulness, LLM judge calibration, and knowledge-graph traversal.

---

## 1. Corpus Overview & Open Licensing

The underlying evaluation documents are situated in `eval/data/` and cover core undergraduate Computer Science curricula (Operating Systems):

| Document | Topic | Length | License |
| :--- | :--- | :--- | :--- |
| `os_process_scheduling.pdf` | Multiprogramming, FCFS, SJF, SRTF, Round Robin, MLFQ, Multiprocessor Affinity, Real-Time (RM, EDF) | 5 pages (1,226 words) | Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) |
| `os_memory_management.pdf` | Logical vs Physical, MMU, Contiguous Allocation, Paging, TLB & EMAT, Virtual Memory, Inverted Tables, Page Faults, LRU, Thrashing | 5 pages (1,202 words) | CC BY-SA 4.0 |
| `os_deadlocks_concurrency.pdf`| Critical Section, Peterson's Algorithm, Mutex, Semaphores, Bounded Buffer, Readers-Writers, Coffman Conditions, Resource Allocation Graphs, Banker's Algorithm | 5 pages (1,259 words) | CC BY-SA 4.0 |

*Total Benchmark Corpus*: 15 structured pages, ~3,700 technical words, 3 core domains.

---

## 2. Dataset Schema (`golden_set.jsonl`)

The dataset contains **65 JSONL items** adhering to the following strict Pydantic schema:

```json
{
  "id": "gold_def_001",
  "question": "What is the definition of CPU scheduling in multiprogramming operating systems?",
  "gold_answer": "CPU scheduling is the basis of multiprogramming operating systems...",
  "gold_pages": [1],
  "gold_chunk_keywords": ["CPU scheduling", "multiprogramming", "productivity", "switches"],
  "difficulty": "easy",
  "type": "definition"
}
```

### Breakdown of Question Typology:
- **`definition` (15 items)**: Formal academic definitions (e.g., TLB, External Fragmentation, Deadlock, Semaphore).
- **`factual` (23 items)**: Direct single-hop concept lookups (e.g., EMAT formula, 50% rule, Coffman conditions, Banker's data structures).
- **`multi-hop` (15 items)**: Cross-page or relational reasoning questions requiring combining 2+ concepts (e.g., how quantum size ties into TLB invalidation; cycle detection in single vs multi-instance RAGs; LRU approximating OPT).
- **`unanswerable` (12 items)**: Queries completely outside the corpus domain (Quantum Computing, Transformer Attention, Roman Architecture, Blockchain) to strictly evaluate **abstention precision** without hallucinated citations.

---

## 3. Ground-Truth Page & Keyword Mapping

- **Page Ground Truth (`gold_pages`)**: Recorded as 1-indexed integers corresponding to the exact document pages containing the factual proof. For multi-hop questions, all necessary pages are included (e.g., `[2, 3]`). For unanswerable questions, `gold_pages` is strictly `[]`.
- **Chunk Keywords (`gold_chunk_keywords`)**: Essential technical terms that must appear in the retrieved passage for it to be counted as factually sufficient.

---

## 4. Methodology & Construction Limitations

1. **Synthetic Academic Curation**: The questions and reference answers were curated to match university examination standards (e.g., JNTU, GATE, ABET CS standards).
2. **Corpus Boundaries**: The corpus is restricted to 15 pages across 3 modules. While ideal for controlled, reproducible evaluation, real enterprise deployments span thousands of pages.
3. **Keyword Heuristics**: Token-level lexical checks (EM/F1) can underestimate semantically valid paraphrases; hence, we couple them with LLM-as-a-judge and human calibration checks.
