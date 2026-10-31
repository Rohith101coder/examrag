"""Unit tests for evaluation metrics, mathematical formulas, and LLM judge parsing.

Tests run entirely in-memory with deterministic known inputs and outputs (no network calls).
"""

import pytest

from eval.generation_eval import (
    JudgeEvaluation,
    compute_exact_match,
    compute_token_f1,
    normalize_text,
)
from eval.retrieval_eval import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


class TestRetrievalMetrics:
    """Mathematical validation for ranking and retrieval metrics."""

    def test_hit_at_k(self):
        gold = [2, 3]
        assert calculate_hit_at_k([1, 2, 4], gold, k=1) == 0.0
        assert calculate_hit_at_k([1, 2, 4], gold, k=2) == 1.0
        assert calculate_hit_at_k([1, 2, 4], gold, k=3) == 1.0
        assert calculate_hit_at_k([], gold, k=3) == 0.0
        assert calculate_hit_at_k([1, 4, 5], gold, k=3) == 0.0

    def test_precision_at_k(self):
        gold = [2, 3]
        # k=1: retrieved [2] -> 1/1 = 1.0
        assert calculate_precision_at_k([2, 4, 5], gold, k=1) == 1.0
        # k=2: retrieved [2, 4] -> 1/2 = 0.5
        assert calculate_precision_at_k([2, 4, 5], gold, k=2) == 0.5
        # k=3: retrieved [2, 3, 5] -> 2/3 = 0.6667
        assert calculate_precision_at_k([2, 3, 5], gold, k=3) == 0.6667
        # Empty
        assert calculate_precision_at_k([], gold, k=3) == 0.0

    def test_recall_at_k(self):
        gold = [2, 3]
        # k=1: hits {2} out of {2, 3} -> 1/2 = 0.5
        assert calculate_recall_at_k([2, 4, 5], gold, k=1) == 0.5
        # k=2: hits {2, 3} out of {2, 3} -> 2/2 = 1.0
        assert calculate_recall_at_k([2, 3, 5], gold, k=2) == 1.0
        # Empty
        assert calculate_recall_at_k([], gold, k=3) == 0.0

    def test_mrr(self):
        gold = [3]
        # First hit at rank 1 -> 1/1 = 1.0
        assert calculate_mrr([3, 1, 2], gold) == 1.0
        # First hit at rank 2 -> 1/2 = 0.5
        assert calculate_mrr([1, 3, 2], gold) == 0.5
        # First hit at rank 3 -> 1/3 = 0.3333
        assert calculate_mrr([1, 2, 3], gold) == 0.3333
        # No hit -> 0.0
        assert calculate_mrr([1, 2, 4], gold) == 0.0

    def test_ndcg_at_k(self):
        gold = [2, 3]
        # Ideal order: both top 2 are relevant
        assert calculate_ndcg_at_k([2, 3, 5], gold, k=2) == 1.0
        assert calculate_ndcg_at_k([3, 2, 5], gold, k=2) == 1.0
        # First relevant at rank 2 -> discounted
        ndcg_delayed = calculate_ndcg_at_k([1, 2, 5], gold, k=2)
        assert 0.0 < ndcg_delayed < 1.0
        # Zero hits
        assert calculate_ndcg_at_k([1, 4, 5], gold, k=3) == 0.0


class TestGenerationTokenMetrics:
    """Exact Match and Token F1 validation."""

    def test_normalize_text(self):
        raw = "The Round-Robin (RR) scheduling algorithm, an optimal choice!"
        normalized = normalize_text(raw)
        assert "the" not in normalized.split()
        assert "an" not in normalized.split()
        assert "," not in normalized
        assert "!" not in normalized

    def test_exact_match(self):
        pred = "Round Robin scheduling uses a time quantum."
        gold = "Round Robin scheduling uses a time quantum!"
        assert compute_exact_match(pred, gold) == 1.0

        diff = "Round Robin is non-preemptive."
        assert compute_exact_match(diff, gold) == 0.0

    def test_token_f1(self):
        pred = "Translation Lookaside Buffer caches page table entries in hardware"
        gold = "Translation Lookaside Buffer caches virtual address translations"
        f1 = compute_token_f1(pred, gold)
        assert 0.4 < f1 < 0.9

        # Identical tokens
        assert compute_token_f1("exact match text", "exact match text") == 1.0
        # Completely disjoint
        assert compute_token_f1("apple banana", "orange grape") == 0.0


class TestJudgeSchemaAndParsing:
    """LLM-as-a-Judge schema integrity and JSON boundary checks."""

    def test_valid_judge_evaluation_schema(self):
        data = {
            "faithfulness": 4.5,
            "answer_relevance": 5.0,
            "correctness_vs_gold": 4.0,
            "citation_accuracy": 5.0,
            "reasoning": "Answer is accurately grounded in retrieved passages with precise citation."
        }
        judge_res = JudgeEvaluation(**data)
        assert judge_res.faithfulness == 4.5
        assert judge_res.reasoning.startswith("Answer is accurately")

    def test_invalid_judge_evaluation_range(self):
        data = {
            "faithfulness": 6.5,  # Exceeds max 5.0
            "answer_relevance": 5.0,
            "correctness_vs_gold": 4.0,
            "citation_accuracy": 5.0,
            "reasoning": "Out of range"
        }
        with pytest.raises(Exception):
            JudgeEvaluation(**data)
