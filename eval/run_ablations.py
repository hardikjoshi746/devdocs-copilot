"""
eval/run_ablations.py

Runs the pipeline against all 50 dataset questions and computes metrics.
Also runs ablation variants to measure the contribution of each technique.

Ablation table:
| Variant                          | Recall@5 | Correctness | Faithfulness | Latency p95 |
|----------------------------------|----------|-------------|--------------|-------------|
| Baseline: dense only, no HyDE    |          |             |              |             |
| + Sparse (hybrid RRF)            |          |             |              |             |
| + Reranker                       |          |             |              |             |
| + HyDE query rewriting           |          |             |              |             |
| Full system                      |          |             |              |             |

Each row isolates one variable. Numbers either justify the technique or cut it.

Usage:
    python -m eval.run_ablations                    # full system only
    python -m eval.run_ablations --ablations        # all variants
    python -m eval.run_ablations --variant baseline # one variant
"""

import asyncio
import json
import time
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ingestion.chunkers import Document
from eval.metrics import recall_at_5, answer_correctness, faithfulness


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------

async def run_baseline(question: str, n_results: int = 5) -> tuple[list[Document], dict]:
    """Dense search only, no HyDE, no reranker."""
    from retrieval.dense import dense_search
    from generation.answer import generate

    docs = await dense_search(question, n_results=n_results)
    result = await generate(question, docs)
    return docs, result


async def run_hybrid(question: str, n_results: int = 5) -> tuple[list[Document], dict]:
    """Dense + sparse (RRF), no HyDE, no reranker."""
    from retrieval.hybrid import hybrid_search
    from generation.answer import generate

    docs = await hybrid_search(question, n_results=n_results)
    result = await generate(question, docs)
    return docs, result


async def run_hybrid_rerank(question: str, n_results: int = 5) -> tuple[list[Document], dict]:
    """Dense + sparse (RRF) + reranker, no HyDE."""
    from retrieval.hybrid import hybrid_search
    from retrieval.reranker import rerank_async
    from generation.answer import generate

    docs = await hybrid_search(question, n_results=20)
    docs = await rerank_async(question, docs, top_n=n_results)
    result = await generate(question, docs)
    return docs, result


async def run_full(question: str, n_results: int = 5) -> tuple[list[Document], dict]:
    """Full system: HyDE + hybrid + reranker + evaluator + faithfulness check."""
    from query.pipeline import pipeline
    from generation.answer import generate
    from evaluator.retrieval_evaluator import evaluate
    from evaluator.faithfulness_check import check_faithfulness

    docs = await pipeline(question)
    quality = await evaluate(question, docs)
    if quality == "ABSTAIN":
        return docs, {"answer": "", "citations": []}
    result = await generate(question, docs)
    filtered = await check_faithfulness(result["answer"], docs)
    return docs, {"answer": filtered, "citations": result["citations"]}


VARIANTS = {
    "baseline": run_baseline,
    "hybrid": run_hybrid,
    "hybrid_rerank": run_hybrid_rerank,
    "full": run_full,
}


# ---------------------------------------------------------------------------
# Evaluate one variant against the full dataset
# ---------------------------------------------------------------------------

async def evaluate_variant(variant_name: str, dataset: list[dict]) -> dict:
    """
    Run a pipeline variant against all dataset questions and compute metrics.

    Returns aggregated metrics for the variant.
    """
    run_fn = VARIANTS[variant_name]
    results = []
    latencies = []

    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"{'='*60}")

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_ids = item.get("expected_source_ids", [])

        print(f"[{i+1}/{len(dataset)}] {item['id']}: {question[:55]}...")

        for attempt in range(3):
            try:
                start = time.perf_counter()
                docs, result = await run_fn(question)
                latency = time.perf_counter() - start
                latencies.append(latency)

                retrieved_ids = [doc.id for doc in docs]
                r5 = recall_at_5(retrieved_ids, expected_ids)

                # Score correctness and faithfulness concurrently
                correctness, faith = await asyncio.gather(
                    answer_correctness(question, result["answer"], expected_answer),
                    faithfulness(result["answer"], docs),
                )

                results.append({
                    "id": item["id"],
                    "category": item["category"],
                    "recall_at_5": r5,
                    "answer_correctness": correctness,
                    "faithfulness": faith,
                    "latency_s": round(latency, 2),
                })

                print(f"  recall@5={r5:.1f} correctness={correctness:.2f} faith={faith:.2f} latency={latency:.1f}s")
                break  # success — exit retry loop

            except Exception as e:
                if attempt < 2:
                    print(f"  attempt {attempt+1} failed: {e} — retrying in 5s...")
                    await asyncio.sleep(5)
                else:
                    print(f"  ERROR (3 attempts): {e}")
                    results.append({
                        "id": item["id"],
                        "category": item["category"],
                        "recall_at_5": 0.0,
                        "answer_correctness": 0.0,
                        "faithfulness": 0.0,
                        "latency_s": 0.0,
                        "error": str(e),
                    })

    # Aggregate
    n = len(results)
    latencies_sorted = sorted(latencies)
    p95_idx = int(0.95 * len(latencies_sorted))
    p95_latency = latencies_sorted[p95_idx] if latencies_sorted else 0.0

    aggregated = {
        "variant": variant_name,
        "n": n,
        "recall_at_5": round(sum(r["recall_at_5"] for r in results) / n, 3),
        "answer_correctness": round(sum(r["answer_correctness"] for r in results) / n, 3),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 3),
        "latency_p95_s": round(p95_latency, 2),
        "per_question": results,
    }

    # Per-category breakdown
    for category in ["factual", "conceptual", "cross-source", "debug"]:
        cat_results = [r for r in results if r.get("category") == category]
        if cat_results:
            aggregated[f"recall_at_5_{category}"] = round(
                sum(r["recall_at_5"] for r in cat_results) / len(cat_results), 3
            )

    return aggregated


# ---------------------------------------------------------------------------
# Print ablation table
# ---------------------------------------------------------------------------

def print_table(all_results: list[dict]) -> None:
    print(f"\n{'='*75}")
    print("ABLATION TABLE")
    print(f"{'='*75}")
    header = f"{'Variant':<20} {'Recall@5':>8} {'Correct':>8} {'Faith':>8} {'Lat p95':>8}"
    print(header)
    print("-" * 55)
    for r in all_results:
        print(
            f"{r['variant']:<20} "
            f"{r['recall_at_5']:>8.3f} "
            f"{r['answer_correctness']:>8.3f} "
            f"{r['faithfulness']:>8.3f} "
            f"{r['latency_p95_s']:>7.1f}s"
        )
    print(f"{'='*75}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(variants_to_run: list[str], limit: int | None = None) -> None:
    # Load dataset
    dataset_path = Path("eval/dataset_with_ids.json")
    if not dataset_path.exists():
        print("dataset_with_ids.json not found.")
        print("Run: python -m eval.dataset")
        return

    with dataset_path.open() as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    print(f"Loaded {len(dataset)} questions from dataset_with_ids.json")

    all_results = []
    for variant_name in variants_to_run:
        result = await evaluate_variant(variant_name, dataset)
        all_results.append(result)

        # Save after each variant in case of interruption
        output = Path(f"eval/results_{variant_name}.json")
        with output.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output}")

    print_table(all_results)

    # Save combined results
    with Path("eval/ablation_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results saved to eval/ablation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Run a single variant (default: full)"
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run all variants for the full ablation table"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (for quick validation)"
    )
    args = parser.parse_args()

    if args.ablations:
        variants = list(VARIANTS.keys())
    elif args.variant:
        variants = [args.variant]
    else:
        variants = ["full"]

    asyncio.run(main(variants, limit=args.limit))