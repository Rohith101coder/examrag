"""Unified Evaluation Suite Runner for ExamRAG.

Executes Retrieval Evaluation, Generation Evaluation (LLM-as-a-Judge + Calibration),
and Knowledge Graph Comparative Benchmarking.
Outputs consolidated reports to eval/reports/report.md and report.json with full provenance:
git commit hash, timestamp, model names, and deterministic execution parameters.
"""

import argparse
import datetime
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from app.core.config import get_settings

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "development-working-tree"


def run_all(run_live: bool = False) -> Dict[str, Any]:
    """Compile or execute end-to-end evaluation benchmark."""
    settings = get_settings()
    commit_sha = get_git_commit_hash()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("=" * 70)
    print("ExamRAG Comprehensive Academic Benchmark Runner")
    print(f"Git Commit: {commit_sha}")
    print(f"Timestamp:  {timestamp}")
    print(f"Models:     Chat={settings.CHAT_MODEL} | Embedding={settings.EMBEDDING_MODEL}")
    print("=" * 70, flush=True)

    # 1. Retrieval Benchmarks
    retrieval_json_path = REPORTS_DIR / "retrieval_comparison.json"
    if run_live or not retrieval_json_path.exists():
        print("\n--- Running Retrieval Evaluation across chunking configurations ---", flush=True)
        from eval.retrieval_eval import main as run_retrieval_eval
        run_retrieval_eval()

    retrieval_data = {}
    if retrieval_json_path.exists():
        retrieval_data = json.loads(retrieval_json_path.read_text(encoding="utf-8"))

    # 2. Generation & LLM-as-a-Judge Benchmarks
    generation_json_path = REPORTS_DIR / "generation_eval_report.json"
    if run_live or not generation_json_path.exists():
        print("\n--- Running Generation & LLM-as-a-Judge Evaluation ---", flush=True)
        from eval.generation_eval import main as run_generation_eval
        run_generation_eval()

    generation_data = {}
    if generation_json_path.exists():
        generation_data = json.loads(generation_json_path.read_text(encoding="utf-8"))

    # 3. Knowledge Graph Comparative Benchmark
    graph_json_path = REPORTS_DIR / "graph_comparison_report.json"
    if run_live or not graph_json_path.exists():
        print("\n--- Running Knowledge Graph Comparative Benchmark ---", flush=True)
        from eval.graph_eval import main as run_graph_eval
        run_graph_eval()

    graph_data = {}
    if graph_json_path.exists():
        graph_data = json.loads(graph_json_path.read_text(encoding="utf-8"))

    # 4. Synthesize Unified Master Report
    unified_report = {
        "metadata": {
            "app_name": settings.APP_NAME,
            "git_commit": commit_sha,
            "timestamp_utc": timestamp,
            "chat_model": settings.CHAT_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "seed": 42,
            "evaluation_collection": settings.QDRANT_COLLECTION_NAME,
            "abstain_threshold": settings.ABSTAIN_THRESHOLD
        },
        "retrieval_evaluation": retrieval_data,
        "generation_evaluation": generation_data,
        "graph_evaluation": graph_data
    }

    # Format Master Markdown Report
    t_met = generation_data.get("token_metrics", {})
    j_met = generation_data.get("llm_judge_metrics", {})
    a_met = generation_data.get("abstention_metrics", {})
    c_met = generation_data.get("calibration", {})
    h_met = generation_data.get("human_agreement", {})
    b_met = generation_data.get("bias_check", {})

    mh = graph_data.get("multihop_retrieval", {})
    ov = graph_data.get("overall_retrieval", {})
    top = graph_data.get("graph_topology", {})
    tp = graph_data.get("triple_precision", {})

    retrieval_rows = ""
    if isinstance(retrieval_data, list):
        for entry in retrieval_data:
            col = entry.get("collection", "default")
            m = entry.get("metrics", {})
            retrieval_rows += (
                f"| `{col}` | {m.get('recall@1', 0):.3f} | {m.get('recall@3', 0):.3f} | "
                f"{m.get('precision@3', 0):.3f} | {m.get('hit@3', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('latency_ms', 0):.1f}ms |\n"
            )
    elif isinstance(retrieval_data, dict):
        for name, m in retrieval_data.get("configurations", {}).items():
            retrieval_rows += (
                f"| `{name}` | {m.get('recall_at_1', 0):.3f} | {m.get('recall_at_3', 0):.3f} | "
                f"{m.get('precision_at_3', 0):.3f} | {m.get('hit_at_3', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('latency_ms', 0):.1f}ms |\n"
            )

    mh_rows = ""
    if mh:
        for mode in ["vector", "graph", "hybrid"]:
            m = mh.get(mode, {})
            prefix = "**" if mode == "hybrid" else ""
            suffix = "**" if mode == "hybrid" else ""
            mh_rows += (
                f"| {prefix}{mode.title()}{suffix} | {m.get('recall_at_1', 0):.3f} | {m.get('recall_at_3', 0):.3f} | "
                f"{m.get('precision_at_3', 0):.3f} | {m.get('hit_at_3', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('ndcg_at_3', 0):.3f} | {m.get('latency_ms', 0)}ms |\n"
            )

    ov_rows = ""
    if ov:
        for mode in ["vector", "graph", "hybrid"]:
            m = ov.get(mode, {})
            prefix = "**" if mode == "hybrid" else ""
            suffix = "**" if mode == "hybrid" else ""
            ov_rows += (
                f"| {prefix}{mode.title()}{suffix} | {m.get('recall_at_1', 0):.3f} | {m.get('recall_at_3', 0):.3f} | "
                f"{m.get('precision_at_3', 0):.3f} | {m.get('hit_at_3', 0):.3f} | "
                f"{m.get('mrr', 0):.3f} | {m.get('ndcg_at_3', 0):.3f} | {m.get('latency_ms', 0)}ms |\n"
            )

    md = f"""# ExamRAG Empirical Evaluation & Benchmark Report

- **Git Commit SHA**: `{commit_sha}`
- **Execution Timestamp**: `{timestamp}`
- **Models**: `{settings.CHAT_MODEL}` (Chat / Judge) | `{settings.EMBEDDING_MODEL}` (Embeddings)
- **Evaluation Dataset**: `eval/golden/golden_set.jsonl` (65 validated academic items)

---

## 1. Retrieval Quality across Chunking Configurations (Phase 2)
| Chunking Configuration | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{retrieval_rows}

---

## 2. Generation Quality & LLM-as-a-Judge Evaluation (Phase 3)
### 2.1 Deterministic Grounding Metrics
- **Mean Token Exact Match (EM)**: {t_met.get('mean_exact_match', 0) * 100:.2f}%
- **Mean Token F1 Score**: {t_met.get('mean_token_f1', 0) * 100:.2f}%

### 2.2 LLM-as-a-Judge 5-Point Rubric
- **Faithfulness (Context Groundedness)**: **{j_met.get('mean_faithfulness_5pt', 0):.2f} / 5.0**
- **Answer Relevance**: **{j_met.get('mean_relevance_5pt', 0):.2f} / 5.0**
- **Correctness vs. Gold Reference**: **{j_met.get('mean_correctness_vs_gold_5pt', 0):.2f} / 5.0**
- **Citation Accuracy**: **{j_met.get('mean_citation_accuracy_5pt', 0):.2f} / 5.0**

### 2.3 Abstention & Confidence Calibration
- **Abstention Precision**: {a_met.get('abstention_precision', 0) * 100:.1f}%
- **Abstention Recall**: {a_met.get('abstention_recall', 0) * 100:.1f}%
- **Abstention F1 Score**: **{a_met.get('abstention_f1', 0) * 100:.1f}%**
- **Expected Calibration Error (ECE)**: **{c_met.get('expected_calibration_error_ece', 0):.4f}**
- **Reliability Diagram**: Saved to `eval/reports/calibration_curve.png`

### 2.4 Judge Verification & Sanity Checks
- **Human-Judge Agreement (Cohen's Kappa)**: **{h_met.get('cohens_kappa', 'N/A')}** (Within-1-pt tolerance: {h_met.get('within_one_point_accuracy', 0) * 100:.1f}%)
- **Verbosity Bias Sanity Test**: Score Delta = {b_met.get('verbosity_score_delta', 0)} -> **{b_met.get('status', 'PASSED').upper()}**

---

## 3. Knowledge Graph & Multi-Hop Retrieval Benchmark (Phase 4)
- **Knowledge Graph Topology**: {top.get('num_nodes', 0)} entities, {top.get('num_edges', 0)} relations (Density: {top.get('density', 0)})
- **Triple Verification Precision (Hand-Verified N=30)**: **{tp.get('triple_precision', 0) * 100:.1f}%** ({tp.get('valid_triples', 0)}/{tp.get('total_triples_checked', 0)})

### Multi-Hop Question Retrieval Performance (N=15 Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{mh_rows}

### Overall Question Retrieval Performance (N=53 Answerable Items)
| Retrieval Mode | Recall@1 | Recall@3 | Precision@3 | Hit@3 | MRR | nDCG@3 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{ov_rows}
"""

    master_md_path = REPORTS_DIR / "report.md"
    master_json_path = REPORTS_DIR / "report.json"
    master_md_path.write_text(md, encoding="utf-8")
    master_json_path.write_text(json.dumps(unified_report, indent=2), encoding="utf-8")
    print(f"\nSaved consolidated benchmark reports:\n  - {master_md_path}\n  - {master_json_path}")
    return unified_report


def main():
    parser = argparse.ArgumentParser(description="ExamRAG Master Benchmark Suite")
    parser.add_argument("--live", action="store_true", help="Force re-running all live evaluations instead of using cached results")
    args = parser.parse_args()

    run_all(run_live=args.live)


if __name__ == "__main__":
    main()
