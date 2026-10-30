# ExamRAG Evaluation Framework

This directory contains the benchmarking dataset, metrics calculation, and evaluation runner for measuring **retrieval quality**, **answer faithfulness**, **abstention accuracy**, and **execution latency** in ExamRAG.

---

## 1. Evaluation Methodology

A reliable RAG pipeline must be evaluated across distinct, decoupled dimensions:

1. **Retrieval Quality**: Did the retriever locate the actual source pages containing the ground truth? (Recall@k, MRR).
2. **Answer Faithfulness**: Is the generated answer grounded strictly in the retrieved context without hallucinating external theories?
3. **Key Technical Fact Coverage**: Does the exam answer contain the core technical terms and definitions required for full exam credit?
4. **Abstention Reliability**: When presented with out-of-domain, unsupported, or adversarial questions, does the model decline to answer rather than fabricating false citations?
5. **System Performance**: Latency breakdown between vector search (`retrieval_latency_ms`) and LLM synthesis (`generation_latency_ms`).

---

## 2. Dataset Schema (`dataset.json`)

Each sample in `dataset.json` contains:

```json
{
  "id": "eval_01",
  "question": "Representative student exam question",
  "document_ids": ["doc_sample_os"],
  "relevant_pages": [1, 2],
  "expected_key_facts": ["term 1", "term 2", "term 3"],
  "reference_answer": "Ground-truth reference summary",
  "is_supported": true
}
```

- **Supported Questions**: Core computer science curriculum topics (e.g., Process Scheduling, Round Robin, Deadlocks, Paging).
- **Unsupported Questions**: Out-of-syllabus questions (e.g., Quantum Computing, Roman Aqueducts) to verify that negative samples trigger clean abstentions.

---

## 3. Metric Definitions

### Recall@k
Measures what proportion of relevant ground-truth document pages appear in the top $k$ retrieved chunks:
$$\text{Recall@}k = \frac{|\text{Retrieved Pages}_{1..k} \cap \text{Relevant Pages}|}{|\text{Relevant Pages}|}$$

### Mean Reciprocal Rank (MRR)
Measures the position of the first relevant chunk in the retrieved ranking:
$$\text{MRR} = \frac{1}{\text{rank}_1}$$

### Fact Coverage
The fraction of required technical keywords present in the generated answer prose:
$$\text{Fact Coverage} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{fact}_i \in \text{Answer})}{N}$$

### Abstention Rate
For queries labeled `is_supported: false`, checks that the model abstained ("The indexed material does not contain sufficient information...") rather than fabricating an answer.

---

## 4. Running the Evaluation

### Synthetic / CI Mode (No external API calls needed)
```bash
python -m evaluation.evaluate --mock
```

### Live Cluster Mode (Runs against Qdrant Cloud and OpenRouter)
Ensure `.env` has valid credentials and indexed documents, then run:
```bash
python -m evaluation.evaluate
```

