"""Evaluation framework for ExamRAG retrieval quality, answer faithfulness, and latency."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from app.api.schemas.chat import ChatRequest, ExamAnswer, SourceChunk
from app.core.config import get_settings
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def calculate_recall_at_k(retrieved_pages: List[int], relevant_pages: List[int], k: int) -> float:
    """Calculate Recall@k given retrieved page numbers and ground-truth relevant pages."""
    if not relevant_pages:
        return 1.0  # Irrelevant query: successful if no pages were needed

    top_k_retrieved = set(retrieved_pages[:k])
    gold_set = set(relevant_pages)
    hits = len(top_k_retrieved.intersection(gold_set))
    return round(hits / len(gold_set), 4)


def calculate_mrr(retrieved_pages: List[int], relevant_pages: List[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for the first relevant retrieved page."""
    if not relevant_pages:
        return 1.0

    gold_set = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in gold_set:
            return round(1.0 / rank, 4)
    return 0.0


def calculate_fact_coverage(answer_text: str, expected_key_facts: List[str]) -> float:
    """Compute proportion of expected technical facts present in generated answer."""
    if not expected_key_facts:
        return 1.0

    text_lower = answer_text.lower()
    matches = sum(1 for fact in expected_key_facts if fact.lower() in text_lower)
    return round(matches / len(expected_key_facts), 4)


def verify_abstention(answer: ExamAnswer) -> bool:
    """Verify whether an answer correctly abstains from hallucinating unsupported facts."""
    abstain_signals = [
        "does not contain sufficient information",
        "lack the specific details",
        "insufficient information",
        "none available from indexed context"
    ]
    combined_answer = (
        f"{answer.definition} {answer.working} {answer.example} "
        f"{' '.join(answer.advantages)} {' '.join(answer.limitations)}"
    ).lower()

    return any(sig in combined_answer for sig in abstain_signals)


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    mock_mode: bool = False
) -> Dict[str, Any]:
    """Execute evaluation over the benchmark dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    dataset: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = get_settings()

    rag_service = None
    if not mock_mode:
        try:
            rag_service = RAGService(settings=settings)
        except Exception as e:
            logger.warning("Could not initialize live RAGService: %s. Falling back to mock evaluator.", str(e))
            mock_mode = True

    results = []
    print("\n" + "=" * 80)
    print(f"ExamRAG Evaluation Runner | Mode: {'MOCK/SYNTHETIC' if mock_mode else 'LIVE CLUSTER'}")
    print(f"Loaded {len(dataset)} evaluation samples.")
    print("=" * 80)

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        doc_ids = item.get("document_ids")
        relevant_pages = item.get("relevant_pages", [])
        expected_facts = item.get("expected_key_facts", [])
        is_supported = item.get("is_supported", True)

        if mock_mode:
            # Deterministic synthetic fixture for CI / automated tests
            if is_supported:
                retrieved_pages = relevant_pages[:3]
                answer = ExamAnswer(
                    definition=f"Definition of {question.split()[2]} in computer systems.",
                    working=" ".join(expected_facts),
                    example=f"Practical example: {expected_facts[0] if expected_facts else 'system call'}.",
                    advantages=["High throughput", "Fairness"],
                    limitations=["Overhead", "Complexity"]
                )
                sources = [
                    SourceChunk(
                        document_id=doc_ids[0] if doc_ids else "doc_1",
                        filename="OperatingSystems.pdf",
                        page_number=p,
                        chunk_index=i,
                        text=f"Passage covering {', '.join(expected_facts)}",
                        relevance_score=0.88 - (i * 0.05)
                    )
                    for i, p in enumerate(retrieved_pages)
                ]
                retrieval_lat = 45.2
                gen_lat = 310.5
            else:
                retrieved_pages = []
                sources = []
                answer = ExamAnswer(
                    definition="The indexed material does not contain sufficient information to answer this question.",
                    working="The retrieved document passages lack the specific details required to explain this topic.",
                    example="None available from indexed context.",
                    advantages=[],
                    limitations=[]
                )
                retrieval_lat = 25.0
                gen_lat = 0.0
        else:
            assert rag_service is not None
            req = ChatRequest(question=question, document_ids=doc_ids, top_k=3)
            resp = rag_service.answer_query(req)
            answer = resp.answer
            sources = resp.sources
            retrieved_pages = [s.page_number for s in sources]
            retrieval_lat = resp.metadata.retrieval_latency_ms
            gen_lat = resp.metadata.generation_latency_ms

        full_answer_prose = (
            f"{answer.definition} {answer.working} {answer.example} "
            f"{' '.join(answer.advantages)} {' '.join(answer.limitations)}"
        )

        recall_k = calculate_recall_at_k(retrieved_pages, relevant_pages, k=3)
        mrr = calculate_mrr(retrieved_pages, relevant_pages)
        fact_cov = calculate_fact_coverage(full_answer_prose, expected_facts) if is_supported else 1.0

        abstention_ok = verify_abstention(answer) if not is_supported else True

        results.append({
            "id": qid,
            "question": question[:45] + "...",
            "supported": is_supported,
            "retrieved_count": len(sources),
            "recall@3": recall_k,
            "mrr": mrr,
            "fact_coverage": fact_cov,
            "abstention_verified": abstention_ok,
            "retrieval_lat_ms": retrieval_lat,
            "generation_lat_ms": gen_lat,
            "total_lat_ms": round(retrieval_lat + gen_lat, 2)
        })

    # Summary table
    print(f"\n{'ID':<8} | {'Supported':<9} | {'Retrieved':<9} | {'Recall@3':<8} | {'MRR':<6} | {'FactCov':<8} | {'Abstain':<7} | {'Total ms':<8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['id']:<8} | {str(r['supported']):<9} | {r['retrieved_count']:<9} | {r['recall@3']:<8.2f} | "
            f"{r['mrr']:<6.2f} | {r['fact_coverage']:<8.2f} | {str(r['abstention_verified']):<7} | {r['total_lat_ms']:<8.1f}"
        )

    # Averages
    supported_results = [r for r in results if r["supported"]]
    unsupported_results = [r for r in results if not r["supported"]]

    mean_recall = sum(r["recall@3"] for r in supported_results) / len(supported_results) if supported_results else 1.0
    mean_mrr = sum(r["mrr"] for r in supported_results) / len(supported_results) if supported_results else 1.0
    mean_fact_cov = sum(r["fact_coverage"] for r in supported_results) / len(supported_results) if supported_results else 1.0
    abstention_rate = (
        sum(1 for r in unsupported_results if r["abstention_verified"]) / len(unsupported_results)
        if unsupported_results else 1.0
    )
    mean_latency = sum(r["total_lat_ms"] for r in results) / len(results)

    print("=" * 80)
    print("Aggregate Evaluation Metrics:")
    print(f"  Supported Queries Mean Recall@3:     {mean_recall * 100:.1f}%")
    print(f"  Supported Queries Mean MRR:          {mean_mrr:.3f}")
    print(f"  Key Technical Fact Coverage:        {mean_fact_cov * 100:.1f}%")
    print(f"  Unsupported Question Abstention Rate: {abstention_rate * 100:.1f}%")
    print(f"  Mean Total Execution Latency:        {mean_latency:.1f} ms")
    print("=" * 80 + "\n")

    return {
        "mean_recall_at_3": round(mean_recall, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_fact_coverage": round(mean_fact_cov, 4),
        "abstention_rate": round(abstention_rate, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "samples_evaluated": len(results),
        "results": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ExamRAG Evaluation Suite")
    parser.add_argument("--mock", action="store_true", help="Run against synthetic test fixtures")
    args = parser.parse_args()

    run_evaluation(mock_mode=args.mock)

