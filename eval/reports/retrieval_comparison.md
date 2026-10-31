# Retrieval Evaluation & Configuration Benchmark

Empirical retrieval evaluation across configurations based on the 65-item golden dataset (53 answerable queries):

| Configuration | Chunk / Overlap | Recall@1 | Recall@3 | Recall@5 | Prec@3 | Hit@3 | MRR | nDCG@3 | nDCG@5 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard (500/100)** | 500 / 100 | 0.887 | 0.981 | 0.991 | 0.811 | 1.000 | 0.981 | 1.682 | 1.908 | 263.8 ms |
| **Fine-Grained (300/50)** | 300 / 50 | 0.906 | 0.991 | 1.000 | 0.880 | 1.000 | 0.991 | 1.800 | 2.192 | 255.0 ms |
| **Broad (750/150)** | 750 / 150 | 0.924 | 1.000 | 1.000 | 0.774 | 1.000 | 1.000 | 1.635 | 1.747 | 272.8 ms |
