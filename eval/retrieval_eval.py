"""Phase 2: Retrieval Evaluation Engine for ExamRAG.

Evaluates semantic search across:
- Recall@k, Precision@k, Hit@k, MRR, nDCG@k for k in {1, 3, 5}
- Configuration comparisons (chunk_size, chunk_overlap, top_k, embedding model)
- Emits markdown comparison table and structured JSON report
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.infrastructure.embeddings import EmbeddingService
from app.infrastructure.vector_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "golden_set.jsonl"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Core Metric Mathematical Functions (Pure, Testable)
# ============================================================================

def calculate_hit_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    """1.0 if at least one gold page is retrieved in top-k, else 0.0."""
    if not gold_pages:
        return 0.0
    k_pages = retrieved_pages[:k]
    return 1.0 if any(p in gold_pages for p in k_pages) else 0.0


def calculate_precision_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    """Proportion of top-k retrieved chunks that belong to gold pages."""
    if not gold_pages or k <= 0:
        return 0.0
    k_pages = retrieved_pages[:k]
    relevant_retrieved = sum(1 for p in k_pages if p in gold_pages)
    return round(relevant_retrieved / k, 4)


def calculate_recall_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    """Proportion of unique gold pages present in top-k retrieved chunks."""
    if not gold_pages:
        return 0.0
    gold_set = set(gold_pages)
    k_pages_set = set(retrieved_pages[:k])
    hits = len(k_pages_set.intersection(gold_set))
    return round(hits / len(gold_set), 4)


def calculate_mrr(retrieved_pages: List[int], gold_pages: List[int]) -> float:
    """Mean Reciprocal Rank of the first relevant chunk."""
    if not gold_pages:
        return 0.0
    gold_set = set(gold_pages)
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_set:
            return round(1.0 / rank, 4)
    return 0.0


def calculate_ndcg_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k with binary relevance."""
    if not gold_pages or k <= 0:
        return 0.0

    gold_set = set(gold_pages)
    k_pages = retrieved_pages[:k]

    # DCG calculation
    dcg = 0.0
    for rank, p in enumerate(k_pages, start=1):
        rel = 1.0 if p in gold_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # IDCG calculation (ideal ranking places relevant items at the top)
    ideal_hits = min(k, len(gold_set))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ============================================================================
# Retrieval Evaluator Engine
# ============================================================================

class RetrievalEvaluator:
    """Evaluates retrieval quality against the golden dataset under various configs."""

    def __init__(self, golden_path: Path = GOLDEN_PATH):
        self.golden_path = golden_path
        self.dataset = self._load_dataset()
        self._query_cache: Dict[str, List[float]] = {}

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_path}")
        items = []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def precompute_query_embeddings(self, embedding_service: EmbeddingService) -> None:
        """Pre-embed all golden questions in batched calls to avoid redundant network roundtrips."""
        missing_queries = [
            item["question"] for item in self.dataset
            if item["question"] not in self._query_cache
        ]
        if missing_queries:
            print(f"Pre-computing embeddings for {len(missing_queries)} queries in batches...", flush=True)
            vectors = embedding_service.embed_documents(missing_queries)
            for q, vec in zip(missing_queries, vectors):
                self._query_cache[q] = vec
            print(f"Pre-computed {len(self._query_cache)} query vectors.", flush=True)

    def prepare_index(
        self,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        settings: Settings
    ) -> None:
        """Index the 3 evaluation PDFs into a dedicated Qdrant collection."""
        print(
            f"Indexing collection '{collection_name}' (chunk_size={chunk_size}, overlap={chunk_overlap})...",
            flush=True
        )
        custom_settings = settings.model_copy(
            update={
                "QDRANT_COLLECTION_NAME": collection_name,
                "CHUNK_SIZE": chunk_size,
                "CHUNK_OVERLAP": chunk_overlap
            }
        )
        vector_store = QdrantVectorStore(settings=custom_settings)
        embedding_service = EmbeddingService(settings=custom_settings)
        ingestion_service = IngestionService(
            settings=custom_settings,
            vector_store=vector_store,
            embedding_service=embedding_service
        )

        pdf_files = list(DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {DATA_DIR} to index.")

        for pdf in pdf_files:
            summary = ingestion_service.ingest_pdf_file(pdf)
            print(f"  - Indexed {pdf.name} -> {summary.chunk_count} chunks ({summary.status})", flush=True)

    def evaluate_config(
        self,
        collection_name: str,
        top_k: int = 5,
        settings: Optional[Settings] = None
    ) -> Dict[str, Any]:
        """Execute retrieval evaluation for answerable queries in golden set."""
        app_settings = (settings or get_settings()).model_copy(
            update={"QDRANT_COLLECTION_NAME": collection_name}
        )
        vector_store = QdrantVectorStore(settings=app_settings)
        embedding_service = EmbeddingService(settings=app_settings)

        # Ensure query vectors are cached
        self.precompute_query_embeddings(embedding_service)

        # Filter answerable queries for ground-truth retrieval metrics
        answerable_items = [item for item in self.dataset if item.get("gold_pages")]
        unanswerable_items = [item for item in self.dataset if not item.get("gold_pages")]

        metrics_sum = {
            "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0,
            "precision@1": 0.0, "precision@3": 0.0, "precision@5": 0.0,
            "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
            "ndcg@1": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0,
            "mrr": 0.0,
            "latency_ms": 0.0
        }

        per_query_results = []
        start_eval_time = time.perf_counter()

        for item in answerable_items:
            q = item["question"]
            gold_pages = item["gold_pages"]
            query_vector = self._query_cache[q]

            t0 = time.perf_counter()
            scored_points = vector_store.search(
                query_vector=query_vector,
                top_k=top_k
            )
            lat_ms = round((time.perf_counter() - t0) * 1000, 2)

            retrieved_pages = [
                pt.payload.get("page_number", 1) for pt in scored_points if pt.payload
            ]

            h1 = calculate_hit_at_k(retrieved_pages, gold_pages, 1)
            h3 = calculate_hit_at_k(retrieved_pages, gold_pages, 3)
            h5 = calculate_hit_at_k(retrieved_pages, gold_pages, 5)

            p1 = calculate_precision_at_k(retrieved_pages, gold_pages, 1)
            p3 = calculate_precision_at_k(retrieved_pages, gold_pages, 3)
            p5 = calculate_precision_at_k(retrieved_pages, gold_pages, 5)

            r1 = calculate_recall_at_k(retrieved_pages, gold_pages, 1)
            r3 = calculate_recall_at_k(retrieved_pages, gold_pages, 3)
            r5 = calculate_recall_at_k(retrieved_pages, gold_pages, 5)

            n1 = calculate_ndcg_at_k(retrieved_pages, gold_pages, 1)
            n3 = calculate_ndcg_at_k(retrieved_pages, gold_pages, 3)
            n5 = calculate_ndcg_at_k(retrieved_pages, gold_pages, 5)

            mrr = calculate_mrr(retrieved_pages, gold_pages)

            metrics_sum["hit@1"] += h1
            metrics_sum["hit@3"] += h3
            metrics_sum["hit@5"] += h5

            metrics_sum["precision@1"] += p1
            metrics_sum["precision@3"] += p3
            metrics_sum["precision@5"] += p5

            metrics_sum["recall@1"] += r1
            metrics_sum["recall@3"] += r3
            metrics_sum["recall@5"] += r5

            metrics_sum["ndcg@1"] += n1
            metrics_sum["ndcg@3"] += n3
            metrics_sum["ndcg@5"] += n5

            metrics_sum["mrr"] += mrr
            metrics_sum["latency_ms"] += lat_ms

            per_query_results.append({
                "id": item["id"],
                "type": item["type"],
                "gold_pages": gold_pages,
                "retrieved_pages": retrieved_pages,
                "hit@3": h3,
                "recall@3": r3,
                "mrr": mrr,
                "ndcg@3": n3,
                "latency_ms": lat_ms
            })

        n_queries = len(answerable_items)
        mean_metrics = {k: round(v / n_queries, 4) for k, v in metrics_sum.items()}

        total_duration = round(time.perf_counter() - start_eval_time, 2)

        return {
            "collection": collection_name,
            "answerable_queries": n_queries,
            "unanswerable_queries": len(unanswerable_items),
            "top_k": top_k,
            "metrics": mean_metrics,
            "eval_duration_seconds": total_duration,
            "per_query_sample": per_query_results[:5]
        }


# ============================================================================
# CLI Execution & Config Comparison
# ============================================================================

def run_retrieval_comparison(configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Run retrieval evaluation across multiple chunking configurations and save report."""
    settings = get_settings()
    evaluator = RetrievalEvaluator()

    if configs is None:
        configs = [
            {"name": "Standard (500/100)", "chunk_size": 500, "chunk_overlap": 100, "top_k": 5, "collection": "eval_retrieval_c500_o100"},
            {"name": "Fine-Grained (300/50)", "chunk_size": 300, "chunk_overlap": 50, "top_k": 5, "collection": "eval_retrieval_c300_o50"},
            {"name": "Broad (750/150)", "chunk_size": 750, "chunk_overlap": 150, "top_k": 5, "collection": "eval_retrieval_c750_o150"}
        ]

    comparison_results = []

    print("\n" + "=" * 90)
    print("ExamRAG Phase 2: Retrieval Evaluation & Configuration Benchmark")
    print(f"Dataset: {evaluator.golden_path.name} ({len(evaluator.dataset)} items)")
    print("=" * 90)

    for cfg in configs:
        col_name = cfg["collection"]
        # Index corpus
        evaluator.prepare_index(
            collection_name=col_name,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            settings=settings
        )

        # Run eval
        res = evaluator.evaluate_config(
            collection_name=col_name,
            top_k=cfg["top_k"],
            settings=settings
        )
        res["config_name"] = cfg["name"]
        res["chunk_size"] = cfg["chunk_size"]
        res["chunk_overlap"] = cfg["chunk_overlap"]
        comparison_results.append(res)

    # Generate Markdown Table
    md_lines = [
        "# Retrieval Evaluation & Configuration Benchmark",
        "",
        "Empirical retrieval evaluation across configurations based on the 65-item golden dataset (53 answerable queries):",
        "",
        "| Configuration | Chunk / Overlap | Recall@1 | Recall@3 | Recall@5 | Prec@3 | Hit@3 | MRR | nDCG@3 | nDCG@5 | Latency (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in comparison_results:
        m = r["metrics"]
        md_lines.append(
            f"| **{r['config_name']}** | {r['chunk_size']} / {r['chunk_overlap']} | "
            f"{m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
            f"{m['precision@3']:.3f} | {m['hit@3']:.3f} | {m['mrr']:.3f} | "
            f"{m['ndcg@3']:.3f} | {m['ndcg@5']:.3f} | {m['latency_ms']:.1f} ms |"
        )

    md_content = "\n".join(md_lines) + "\n"
    print("\n" + md_content)

    # Save outputs
    md_path = REPORTS_DIR / "retrieval_comparison.md"
    json_path = REPORTS_DIR / "retrieval_comparison.json"

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    print(f"Saved reports:\n  - {md_path}\n  - {json_path}")
    return {"results": comparison_results, "markdown": md_content}


def main():
    parser = argparse.ArgumentParser(description="ExamRAG Retrieval Evaluation Engine")
    parser.add_argument("--compare", action="store_true", help="Compare multiple chunking configs")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in chars")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in chars")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    if args.compare:
        run_retrieval_comparison()
    else:
        cfg = [{
            "name": f"Config ({args.chunk_size}/{args.chunk_overlap})",
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "collection": f"eval_retrieval_c{args.chunk_size}_o{args.chunk_overlap}"
        }]
        run_retrieval_comparison(cfg)


if __name__ == "__main__":
    main()
