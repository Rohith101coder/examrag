# Knowledge Graph vs. Vector vs. Hybrid Comparative Evaluation

## 1. Graph Topology Summary
- **Nodes**: 40 entities
- **Edges**: 38 directed semantic relations
- **Graph Density**: 0.0244
- **Average Node Degree**: 1.9
- **Triple Verification Precision (Hand-Verified N=30)**: **100.0%** (30/30)

---

## 2. Multi-Hop Question Retrieval Benchmark (N=15 Multi-Hop Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@1 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vector (Qdrant)** | 0.733 | 0.933 | 0.933 | 1.000 | 1.000 | 1.000 | 1.583 | 398.35ms |
| **Graph Only** | 0.733 | 0.933 | 0.933 | 1.000 | 1.000 | 1.000 | 1.583 | 436.37ms |
| **Hybrid (RRF)** | **0.600** | **0.933** | **0.533** | **0.867** | **1.000** | **0.933** | **0.963** | **856.9ms** |

---

## 3. Overall Retrieval Benchmark (N=53 Answerable Golden Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@1 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vector (Qdrant)** | 0.887 | 0.981 | 0.811 | 0.962 | 1.000 | 0.981 | 1.682 | 365.26ms |
| **Graph Only** | 0.849 | 0.962 | 0.799 | 0.924 | 0.981 | 0.958 | 1.642 | 336.17ms |
| **Hybrid (RRF)** | **0.679** | **0.981** | **0.390** | **0.755** | **1.000** | **0.871** | **0.908** | **696.18ms** |
