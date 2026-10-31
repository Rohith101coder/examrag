# Generation Evaluation & LLM-as-a-Judge Benchmark

## 1. Token-Level Grounding Metrics
- **Mean Exact Match (EM)**: 0.00%
- **Mean Token F1 Score**: 9.82%

## 2. LLM-as-a-Judge Rubric Evaluation (5-Point Scale)
- **Faithfulness (Context Groundedness)**: **4.97 / 5.0**
- **Answer Relevance**: **4.95 / 5.0**
- **Correctness vs Reference Answer**: **4.97 / 5.0**
- **Citation Accuracy**: **4.25 / 5.0**

## 3. Abstention & Calibration
- **Abstention Precision**: 34.3%
- **Abstention Recall**: 100.0%
- **Abstention F1 Score**: **51.1%**
- **Expected Calibration Error (ECE)**: **0.3386**
- **Calibration Diagram**: Saved to `eval/reports/calibration_curve.png`

## 4. Judge-Quality & Sanity Checks
- **Human-Judge Cohen's Kappa**: **0.0** (Exact Agreement: 95.0%, Within-1-pt: 100.0%)
- **Verbosity Bias Sanity Check**: Score delta = 0.0 -> **PASSED**
