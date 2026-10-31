# ExamRAG Empirical Evaluation & Benchmark Report

- **Git Commit SHA**: `9c02b9db4cb1a62ec1c05614d3165d75481f28b1`
- **Execution Timestamp**: `2026-10-31T16:00:04.372265+00:00`
- **Models**: `openai/gpt-4o-mini` (Chat / Judge) | `text-embedding-3-small` (Embeddings)
- **Evaluation Dataset**: `eval/golden/golden_set.jsonl` (65 validated academic items)

---

## 1. Retrieval Quality across Chunking Configurations (Phase 2)
| Chunking Configuration | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `eval_retrieval_c500_o100` | 0.887 | 0.981 | 0.811 | 1.000 | 0.981 | 263.8ms |
| `eval_retrieval_c300_o50` | 0.906 | 0.991 | 0.880 | 1.000 | 0.991 | 255.0ms |
| `eval_retrieval_c750_o150` | 0.924 | 1.000 | 0.774 | 1.000 | 1.000 | 272.8ms |


---

## 2. Generation Quality & LLM-as-a-Judge Evaluation (Phase 3)
### 2.1 Deterministic Grounding Metrics
- **Mean Token Exact Match (EM)**: 0.00%
- **Mean Token F1 Score**: 9.82%

### 2.2 LLM-as-a-Judge 5-Point Rubric
- **Faithfulness (Context Groundedness)**: **4.97 / 5.0**
- **Answer Relevance**: **4.95 / 5.0**
- **Correctness vs. Gold Reference**: **4.97 / 5.0**
- **Citation Accuracy**: **4.25 / 5.0**

### 2.3 Abstention & Confidence Calibration
- **Abstention Precision**: 34.3%
- **Abstention Recall**: 100.0%
- **Abstention F1 Score**: **51.1%**
- **Expected Calibration Error (ECE)**: **0.3386**
- **Reliability Diagram**: Saved to `eval/reports/calibration_curve.png`

### 2.4 Judge Verification & Sanity Checks
- **Human-Judge Agreement (Cohen's Kappa)**: **0.0** (Within-1-pt tolerance: 100.0%)
- **Verbosity Bias Sanity Test**: Score Delta = 0.0 -> **PASSED**

---

## 3. Knowledge Graph & Multi-Hop Retrieval Benchmark (Phase 4)
- **Knowledge Graph Topology**: 40 entities, 38 relations (Density: 0.0244)
- **Triple Verification Precision (Hand-Verified N=30)**: **100.0%** (30/30)

### Multi-Hop Question Retrieval Performance (N=15 Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vector | 0.733 | 0.933 | 0.933 | 1.000 | 1.000 | 1.583 | 398.35ms |
| Graph | 0.733 | 0.933 | 0.933 | 1.000 | 1.000 | 1.583 | 436.37ms |
| **Hybrid** | 0.600 | 0.933 | 0.533 | 1.000 | 0.933 | 0.963 | 856.9ms |


### Overall Question Retrieval Performance (N=53 Answerable Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vector | 0.887 | 0.981 | 0.811 | 1.000 | 0.981 | 1.682 | 365.26ms |
| Graph | 0.849 | 0.962 | 0.799 | 0.981 | 0.958 | 1.642 | 336.17ms |
| **Hybrid** | 0.679 | 0.981 | 0.390 | 1.000 | 0.871 | 0.908 | 696.18ms |

