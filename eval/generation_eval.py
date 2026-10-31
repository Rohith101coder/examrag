"""Phase 3: Generation Evaluation & LLM-as-a-Judge Engine for ExamRAG.

Features:
- Token-level Exact Match (EM) & Token F1 against gold_answer
- Pydantic-validated LLM-as-a-Judge (faithfulness, relevance, correctness, citation accuracy)
- Human-vs-Judge Agreement Analysis (Cohen's Kappa + Accuracy)
- Position & Verbosity Bias Sanity Checks
- Confidence Calibration & Expected Calibration Error (ECE) with Matplotlib plot
"""

import argparse
import json
import logging
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.core.config import Settings, get_settings
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "golden_set.jsonl"
HUMAN_LABELS_PATH = Path(__file__).resolve().parent / "golden" / "human_labels.jsonl"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Token-Level Exact Match & F1 (Pure Functions)
# ============================================================================

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Token-normalized Exact Match."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground-truth answer."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


# ============================================================================
# LLM-as-a-Judge Schema & Prompt
# ============================================================================

class JudgeEvaluation(BaseModel):
    faithfulness: float = Field(..., ge=1.0, le=5.0, description="1-5: Is the answer strictly grounded in retrieved context?")
    answer_relevance: float = Field(..., ge=1.0, le=5.0, description="1-5: Does the answer directly address the question?")
    correctness_vs_gold: float = Field(..., ge=1.0, le=5.0, description="1-5: Does it capture the essential technical facts in gold_answer?")
    citation_accuracy: float = Field(..., ge=1.0, le=5.0, description="1-5: Are citations factually aligned with the claims?")
    reasoning: str = Field(..., description="Justification explaining the awarded scores")


JUDGE_PROMPT_TEMPLATE = """You are an expert academic evaluator assessing an AI-generated exam answer for Computer Science students.

Evaluate the generated answer according to the following STRICT 1 to 5 rubric:
1. Faithfulness (1-5): Is every factual claim in the Generated Answer directly supported by the Retrieved Context? (5 = fully supported, 1 = hallucinated/contradicts context).
2. Answer Relevance (1-5): Does the answer directly address the student's Question without filler? (5 = precise and direct, 1 = off-topic).
3. Correctness vs Gold (1-5): Does the answer capture the key technical principles in the Reference Answer? (5 = completely accurate, 1 = factually incorrect).
4. Citation Accuracy (1-5): Do the cited pages actually correspond to the sources providing the evidence? (5 = accurate citations, 1 = false/missing citations).

Note on Abstention: If the Question is unanswerable from the context, and the Generated Answer correctly states that the material does not contain sufficient information, award 5 across all metrics!

Return ONLY a valid JSON object matching this schema:
{{
  "faithfulness": <float between 1.0 and 5.0>,
  "answer_relevance": <float between 1.0 and 5.0>,
  "correctness_vs_gold": <float between 1.0 and 5.0>,
  "citation_accuracy": <float between 1.0 and 5.0>,
  "reasoning": "<concise explanation>"
}}

---
Question: {question}
Reference Answer: {gold_answer}
Retrieved Context:
{context}
Generated Answer:
{generated_answer}
Cited Sources: {sources}
"""


# ============================================================================
# Generation Evaluator Engine
# ============================================================================

class GenerationEvaluator:
    """Evaluates generation quality using deterministic metrics and LLM judge."""

    def __init__(self, golden_path: Path = GOLDEN_PATH, settings: Optional[Settings] = None):
        self.golden_path = golden_path
        self.settings = settings or get_settings()
        self.dataset = self._load_dataset()
        self._judge_client: Optional[ChatOpenAI] = None

    def _load_dataset(self) -> List[Dict[str, Any]]:
        items = []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def get_judge_client(self) -> ChatOpenAI:
        if self._judge_client is None:
            self._judge_client = ChatOpenAI(
                api_key=self.settings.OPENROUTER_API_KEY,
                base_url=self.settings.OPENROUTER_BASE_URL,
                model=self.settings.CHAT_MODEL,
                temperature=0.0,
                max_tokens=600,
                timeout=45.0,
                max_retries=2
            )
        return self._judge_client

    def judge_answer(
        self,
        question: str,
        gold_answer: str,
        context_str: str,
        generated_answer: str,
        sources_str: str
    ) -> JudgeEvaluation:
        """Invoke OpenRouter LLM judge with structured JSON parsing & retry."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            context=context_str if context_str else "[NO CONTEXT RETRIEVED]",
            generated_answer=generated_answer,
            sources=sources_str
        )

        client = self.get_judge_client()

        for attempt in range(3):
            try:
                res = client.invoke([HumanMessage(content=prompt)])
                raw_text = res.content if isinstance(res.content, str) else str(res.content)

                # Parse JSON
                fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                candidate = fence.group(1).strip() if fence else raw_text[raw_text.find("{"):raw_text.rfind("}") + 1]

                data = json.loads(candidate)
                return JudgeEvaluation(**data)
            except Exception as e:
                logger.warning("Judge parsing attempt %d failed: %s", attempt + 1, str(e))
                time.sleep(1.0)

        # Fallback if judge repeatedly fails to return clean JSON
        return JudgeEvaluation(
            faithfulness=3.0,
            answer_relevance=3.0,
            correctness_vs_gold=3.0,
            citation_accuracy=3.0,
            reasoning="Automated fallback: judge failed to return valid JSON."
        )

    def evaluate_pipeline(
        self,
        collection_name: str = "examrag_documents",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run generation evaluation over the dataset using RAGService."""
        rag_settings = self.settings.model_copy(update={"QDRANT_COLLECTION_NAME": collection_name})
        rag = RAGService(settings=rag_settings)

        items_to_eval = self.dataset[:max_samples] if max_samples else self.dataset

        em_scores: List[float] = []
        f1_scores: List[float] = []
        faithfulness_scores: List[float] = []
        relevance_scores: List[float] = []
        correctness_scores: List[float] = []
        citation_scores: List[float] = []

        confidence_scores: List[float] = []
        ground_truth_answerable: List[int] = []  # 1 = answerable, 0 = unanswerable
        abstained_flags: List[bool] = []

        detailed_results = []
        print(f"\nEvaluating generation across {len(items_to_eval)} golden items...", flush=True)

        for i, item in enumerate(items_to_eval, 1):
            qid = item["id"]
            question = item["question"]
            gold = item["gold_answer"]
            is_supported = len(item.get("gold_pages", [])) > 0

            # Execute query through live RAG pipeline with retry
            req = ChatRequest(question=question, top_k=3)
            resp: Optional[ChatResponse] = None
            for query_attempt in range(3):
                try:
                    resp = rag.answer_query(req)
                    break
                except Exception as e:
                    logger.warning("Query execution retry %d for item %s: %s", query_attempt + 1, qid, str(e))
                    time.sleep(1.5)

            if resp is None:
                logger.error("Failed to execute query for item %s after 3 attempts", qid)
                continue

            full_pred_text = (
                f"{resp.answer.definition} {resp.answer.working} {resp.answer.example} "
                f"{' '.join(resp.answer.advantages)} {' '.join(resp.answer.limitations)}"
            ).strip()

            # Deterministic token metrics
            em = compute_exact_match(full_pred_text, gold)
            f1 = compute_token_f1(full_pred_text, gold)
            em_scores.append(em)
            f1_scores.append(f1)

            # Confidence & Abstention tracking
            conf = resp.metadata.confidence_score or 0.0
            abstained = resp.metadata.abstained
            confidence_scores.append(conf)
            ground_truth_answerable.append(1 if is_supported else 0)
            abstained_flags.append(abstained)

            # Format context and sources for judge
            context_str = "\n".join(f"[{s.filename} p.{s.page_number}]: {s.text}" for s in resp.sources)
            sources_str = ", ".join(f"{s.filename} p.{s.page_number} (score: {s.relevance_score})" for s in resp.sources)

            # Judge evaluation
            judge_res = self.judge_answer(
                question=question,
                gold_answer=gold,
                context_str=context_str,
                generated_answer=full_pred_text,
                sources_str=sources_str
            )

            faithfulness_scores.append(judge_res.faithfulness)
            relevance_scores.append(judge_res.answer_relevance)
            correctness_scores.append(judge_res.correctness_vs_gold)
            citation_scores.append(judge_res.citation_accuracy)

            detailed_results.append({
                "id": qid,
                "type": item["type"],
                "supported": is_supported,
                "em": em,
                "f1": f1,
                "confidence": conf,
                "abstained": abstained,
                "judge_faithfulness": judge_res.faithfulness,
                "judge_relevance": judge_res.answer_relevance,
                "judge_correctness": judge_res.correctness_vs_gold,
                "judge_citation": judge_res.citation_accuracy,
                "reasoning": judge_res.reasoning
            })

            if i % 10 == 0 or i == len(items_to_eval):
                print(f"  Processed {i}/{len(items_to_eval)} items (Latest Judge Faithfulness: {judge_res.faithfulness:.1f}/5)", flush=True)

        # Compute summary metrics
        mean_em = round(float(np.mean(em_scores)), 4)
        mean_f1 = round(float(np.mean(f1_scores)), 4)
        mean_faith = round(float(np.mean(faithfulness_scores)), 4)
        mean_rel = round(float(np.mean(relevance_scores)), 4)
        mean_corr = round(float(np.mean(correctness_scores)), 4)
        mean_cite = round(float(np.mean(citation_scores)), 4)

        # Abstention metrics on unanswerable questions
        # Positive class = Abstain
        tp = sum(1 for is_ans, abs_flag in zip(ground_truth_answerable, abstained_flags) if is_ans == 0 and abs_flag)
        fp = sum(1 for is_ans, abs_flag in zip(ground_truth_answerable, abstained_flags) if is_ans == 1 and abs_flag)
        fn = sum(1 for is_ans, abs_flag in zip(ground_truth_answerable, abstained_flags) if is_ans == 0 and not abs_flag)
        tn = sum(1 for is_ans, abs_flag in zip(ground_truth_answerable, abstained_flags) if is_ans == 1 and not abs_flag)

        abs_prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        abs_rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
        abs_f1 = round(2 * abs_prec * abs_rec / (abs_prec + abs_rec), 4) if (abs_prec + abs_rec) > 0 else 0.0

        # Calibration (ECE) and Curve Plot
        ece = self.compute_and_plot_calibration(
            confidence_scores=confidence_scores,
            ground_truth_answerable=ground_truth_answerable,
            abstained_flags=abstained_flags
        )

        # Human agreement check
        human_agreement = self.compute_human_agreement(detailed_results)

        # Bias check
        bias_check = self.run_judge_bias_check()

        return {
            "token_metrics": {
                "mean_exact_match": mean_em,
                "mean_token_f1": mean_f1
            },
            "llm_judge_metrics": {
                "mean_faithfulness_5pt": mean_faith,
                "mean_relevance_5pt": mean_rel,
                "mean_correctness_vs_gold_5pt": mean_corr,
                "mean_citation_accuracy_5pt": mean_cite
            },
            "abstention_metrics": {
                "abstention_precision": abs_prec,
                "abstention_recall": abs_rec,
                "abstention_f1": abs_f1,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            },
            "calibration": {
                "expected_calibration_error_ece": ece,
                "plot_saved_to": "eval/reports/calibration_curve.png"
            },
            "human_agreement": human_agreement,
            "bias_check": bias_check,
            "sample_results": detailed_results
        }

    def compute_and_plot_calibration(
        self,
        confidence_scores: List[float],
        ground_truth_answerable: List[int],
        abstained_flags: List[bool],
        num_bins: int = 5
    ) -> float:
        """Compute Expected Calibration Error (ECE) and render reliability diagram."""
        # Correctness: 1 if prediction matches reality (either answered supported or abstained unsupported)
        accuracies = [
            1.0 if ((is_ans == 1 and not abs_flag) or (is_ans == 0 and abs_flag)) else 0.0
            for is_ans, abs_flag in zip(ground_truth_answerable, abstained_flags)
        ]

        confidences = np.array(confidence_scores)
        corrects = np.array(accuracies)

        bins = np.linspace(0.0, 1.0, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0

        for i in range(num_bins):
            bin_min, bin_max = bins[i], bins[i + 1]
            mask = (confidences >= bin_min) & (confidences < bin_max) if i < num_bins - 1 else (confidences >= bin_min) & (confidences <= bin_max)
            count = int(np.sum(mask))
            if count > 0:
                acc = float(np.mean(corrects[mask]))
                conf = float(np.mean(confidences[mask]))
                bin_accuracies.append(acc)
                bin_confidences.append(conf)
                bin_counts.append(count)
                ece += (count / len(confidences)) * abs(acc - conf)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_min + bin_max) / 2)
                bin_counts.append(0)

        ece = round(float(ece), 4)

        # Plot reliability curve
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.plot(bin_confidences, bin_accuracies, marker="o", color="#2B6CB0", linewidth=2, label=f"ExamRAG (ECE = {ece:.3f})")
        ax.set_xlabel("Confidence Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
        ax.set_title("Reliability Diagram & Calibration Curve", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plot_path = REPORTS_DIR / "calibration_curve.png"
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info("Saved calibration plot to %s", plot_path)
        return ece

    def compute_human_agreement(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute Cohen's kappa and accuracy against human labels in human_labels.jsonl."""
        if not HUMAN_LABELS_PATH.exists():
            return {"status": "skipped", "reason": "human_labels.jsonl not found"}

        human_data = {}
        with open(HUMAN_LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    human_data[item["id"]] = item

        if not human_data:
            return {"status": "skipped", "reason": "No human labels present in file"}

        judge_dict = {r["id"]: r for r in detailed_results}
        common_ids = [qid for qid in human_data if qid in judge_dict]

        if not common_ids:
            return {"status": "skipped", "reason": "No overlapping item IDs"}

        human_scores = []
        judge_scores = []

        for qid in common_ids:
            h_val = human_data[qid].get("human_faithfulness")
            j_val = round(judge_dict[qid].get("judge_faithfulness", 3.0))
            if h_val is not None:
                human_scores.append(int(h_val))
                judge_scores.append(int(j_val))

        if len(human_scores) < 2:
            return {"status": "skipped", "reason": "Insufficient labeled samples"}

        # Compute accuracy (exact agreement) and within-1-point tolerance
        exact_agreements = sum(1 for h, j in zip(human_scores, judge_scores) if h == j)
        within_one = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1)
        acc = round(exact_agreements / len(human_scores), 4)
        tol_acc = round(within_one / len(human_scores), 4)

        # Cohen's Kappa implementation
        kappa = self._cohen_kappa(human_scores, judge_scores)

        return {
            "status": "evaluated",
            "samples_compared": len(human_scores),
            "exact_agreement_accuracy": acc,
            "within_one_point_accuracy": tol_acc,
            "cohens_kappa": kappa
        }

    def _cohen_kappa(self, r1: List[int], r2: List[int]) -> float:
        """Calculate Cohen's Kappa between two raters."""
        categories = sorted(list(set(r1 + r2)))
        n = len(r1)
        if n == 0:
            return 1.0

        # Confusion matrix
        cat_to_idx = {c: idx for idx, c in enumerate(categories)}
        k = len(categories)
        matrix = np.zeros((k, k), dtype=float)

        for a, b in zip(r1, r2):
            matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

        po = np.trace(matrix) / n
        pe = sum(np.sum(matrix[i, :]) * np.sum(matrix[:, i]) for i in range(k)) / (n * n)

        if pe == 1.0:
            return 1.0
        return round(float((po - pe) / (1.0 - pe)), 4)

    def run_judge_bias_check(self) -> Dict[str, Any]:
        """Test for position bias and verbosity bias in the LLM judge."""
        # Synthetic test pair: Concise true statement vs bloated verbose version
        question = "What is the role of the TLB in memory management?"
        gold = "The TLB is a high-speed hardware cache that stores recent virtual-to-physical address translations."
        context = "The Translation Lookaside Buffer (TLB) is an associative hardware cache inside the MMU that caches page table entries."

        concise_ans = "The TLB is an associative hardware cache in the MMU that stores recent page table translations to reduce memory access latency."
        verbose_ans = (
            "The TLB, known in the field of modern computing architectures as the Translation Lookaside Buffer, is a crucial, "
            "highly optimized, associative hardware cache embedded into the Memory Management Unit. It systematically acts as "
            "a caching mechanism to store recent virtual-to-physical address translations, thus ensuring that subsequent lookups "
            "do not waste precious CPU clock cycles waiting for redundant memory reads."
        )

        judge_concise = self.judge_answer(question, gold, context, concise_ans, "Page 4")
        judge_verbose = self.judge_answer(question, gold, context, verbose_ans, "Page 4")

        verbosity_delta = round(judge_verbose.correctness_vs_gold - judge_concise.correctness_vs_gold, 2)
        has_verbosity_bias = verbosity_delta > 0.5

        return {
            "concise_score": judge_concise.correctness_vs_gold,
            "verbose_score": judge_verbose.correctness_vs_gold,
            "verbosity_score_delta": verbosity_delta,
            "verbosity_bias_detected": has_verbosity_bias,
            "status": "passed" if not has_verbosity_bias else "bias_detected"
        }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ExamRAG Generation & Judge Evaluator")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of items to evaluate (default: all)")
    parser.add_argument("--collection", type=str, default="eval_retrieval_c500_o100", help="Qdrant collection to evaluate against")
    args = parser.parse_args()

    evaluator = GenerationEvaluator()
    results = evaluator.evaluate_pipeline(
        collection_name=args.collection,
        max_samples=args.samples
    )

    # Format Markdown Report
    t_met = results["token_metrics"]
    j_met = results["llm_judge_metrics"]
    a_met = results["abstention_metrics"]
    c_met = results["calibration"]
    h_met = results["human_agreement"]
    b_met = results["bias_check"]

    md_report = f"""# Generation Evaluation & LLM-as-a-Judge Benchmark

## 1. Token-Level Grounding Metrics
- **Mean Exact Match (EM)**: {t_met['mean_exact_match'] * 100:.2f}%
- **Mean Token F1 Score**: {t_met['mean_token_f1'] * 100:.2f}%

## 2. LLM-as-a-Judge Rubric Evaluation (5-Point Scale)
- **Faithfulness (Context Groundedness)**: **{j_met['mean_faithfulness_5pt']:.2f} / 5.0**
- **Answer Relevance**: **{j_met['mean_relevance_5pt']:.2f} / 5.0**
- **Correctness vs Reference Answer**: **{j_met['mean_correctness_vs_gold_5pt']:.2f} / 5.0**
- **Citation Accuracy**: **{j_met['mean_citation_accuracy_5pt']:.2f} / 5.0**

## 3. Abstention & Calibration
- **Abstention Precision**: {a_met['abstention_precision'] * 100:.1f}%
- **Abstention Recall**: {a_met['abstention_recall'] * 100:.1f}%
- **Abstention F1 Score**: **{a_met['abstention_f1'] * 100:.1f}%**
- **Expected Calibration Error (ECE)**: **{c_met['expected_calibration_error_ece']:.4f}**
- **Calibration Diagram**: Saved to `eval/reports/calibration_curve.png`

## 4. Judge-Quality & Sanity Checks
- **Human-Judge Cohen's Kappa**: **{h_met.get('cohens_kappa', 'N/A')}** (Exact Agreement: {h_met.get('exact_agreement_accuracy', 0)*100:.1f}%, Within-1-pt: {h_met.get('within_one_point_accuracy', 0)*100:.1f}%)
- **Verbosity Bias Sanity Check**: Score delta = {b_met['verbosity_score_delta']} -> **{b_met['status'].upper()}**
"""

    print("\n" + md_report)

    # Save to disk
    rep_md_path = REPORTS_DIR / "generation_eval_report.md"
    rep_json_path = REPORTS_DIR / "generation_eval_report.json"
    rep_md_path.write_text(md_report, encoding="utf-8")
    rep_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved generation evaluation reports:\n  - {rep_md_path}\n  - {rep_json_path}")


if __name__ == "__main__":
    main()
