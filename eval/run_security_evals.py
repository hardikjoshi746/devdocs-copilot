"""
eval/run_security_evals.py

Runs two security-focused evaluations against the live LLM pipeline.

Eval 1 — Injection Robustness
  For each case in INJECTION_DATASET:
    1. Create a Document whose content is the poisoned chunk.
    2. Call generate(question, [poisoned_doc]) — real Claude call.
    3. Score with injection_robustness() — string match + LLM judge.
  Reports: pass rate per injection style and overall.

Eval 2 — PII Leakage Rate
  For each case in PII_DATASET:
    1. Create a Document whose content is the PII-seeded chunk.
    2. Call generate(question, [pii_doc]) — real Claude call.
    3. Apply mask_pii(answer) — the production masking pipeline.
    4. Score with pii_leakage_rate() — exact string scan.
  Reports: leak rate per PII type and overall.

Usage:
    python -m eval.run_security_evals                # both evals
    python -m eval.run_security_evals --eval injection
    python -m eval.run_security_evals --eval pii
    python -m eval.run_security_evals --limit 5      # quick smoke test

Cost estimate: ~30 Haiku calls (judge) + 30 Sonnet calls (generate) ≈ $0.10
"""

import asyncio
import json
import argparse
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ingestion.chunkers import Document
from generation.answer import generate
from api.guardrails import mask_pii
from eval.security_dataset import INJECTION_DATASET, PII_DATASET
from eval.security_metrics import injection_robustness, pii_leakage_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(chunk_id: str, content: str) -> Document:
    return Document(
        id=chunk_id,
        content=content,
        type="doc",
        source="security_eval",
        parent_id=None,
        metadata={"repo": "security_eval"},
    )


# ---------------------------------------------------------------------------
# Injection robustness eval
# ---------------------------------------------------------------------------

async def run_injection_eval(dataset: list[dict]) -> dict:
    """
    Run all injection test cases and return aggregated results.

    For each case:
    - Build a Document with the poisoned chunk content
    - Call generate() — exercises the system prompt hardening + XML delimiters
    - Score with injection_robustness()
    """
    results = []
    print(f"\n{'='*65}")
    print("EVAL 1 — Injection Robustness")
    print(f"{'='*65}")
    print(f"Running {len(dataset)} test cases...\n")

    for i, case in enumerate(dataset):
        doc = _make_doc(f"inj_eval::{case['id']}", case["poisoned_chunk"])

        for attempt in range(3):
            try:
                start = time.perf_counter()
                result = await generate(case["question"], [doc])
                answer = result["answer"]
                latency = time.perf_counter() - start

                score = await injection_robustness(
                    question=case["question"],
                    generated_answer=answer,
                    injected_instruction=case["injection_indicator"],
                    injection_indicator=case["injection_indicator"],
                )

                status = "SAFE" if score == 1.0 else "HIJACKED"
                print(
                    f"[{i+1:02d}/{len(dataset)}] {case['id']} ({case['style']:<24}) "
                    f"{status:<8} {latency:.1f}s"
                )
                if score == 0.0:
                    print(f"         answer preview: {answer[:120]!r}")

                results.append({
                    "id": case["id"],
                    "style": case["style"],
                    "score": score,
                    "latency_s": round(latency, 2),
                    "answer_preview": answer[:200],
                })
                break

            except Exception as e:
                if attempt < 2:
                    print(f"  attempt {attempt+1} failed: {e} — retrying in 5s...")
                    await asyncio.sleep(5)
                else:
                    print(f"  [{case['id']}] ERROR after 3 attempts: {e}")
                    results.append({
                        "id": case["id"],
                        "style": case["style"],
                        "score": 0.0,
                        "error": str(e),
                    })

    # Aggregate
    n = len(results)
    pass_count = sum(1 for r in results if r["score"] == 1.0)
    overall_rate = pass_count / n if n > 0 else 0.0

    # Per-style breakdown
    styles = sorted({r["style"] for r in results})
    style_rates = {}
    for style in styles:
        style_results = [r for r in results if r["style"] == style]
        style_pass = sum(1 for r in style_results if r["score"] == 1.0)
        style_rates[style] = style_pass / len(style_results)

    return {
        "eval": "injection_robustness",
        "n": n,
        "pass_count": pass_count,
        "pass_rate": round(overall_rate, 3),
        "per_style": {s: round(v, 3) for s, v in style_rates.items()},
        "per_case": results,
    }


# ---------------------------------------------------------------------------
# PII leakage rate eval
# ---------------------------------------------------------------------------

async def run_pii_eval(dataset: list[dict]) -> dict:
    """
    Run all PII test cases and return aggregated results.

    For each case:
    - Build a Document with the PII-seeded chunk content
    - Call generate() — LLM may quote or paraphrase the PII
    - Apply mask_pii() — production Presidio masking
    - Score with pii_leakage_rate() — exact string scan for seeded entities
    """
    results = []
    print(f"\n{'='*65}")
    print("EVAL 2 — PII Leakage Rate")
    print(f"{'='*65}")
    print(f"Running {len(dataset)} test cases...\n")

    for i, case in enumerate(dataset):
        doc = _make_doc(f"pii_eval::{case['id']}", case["pii_chunk"])

        for attempt in range(3):
            try:
                start = time.perf_counter()
                result = await generate(case["question"], [doc])
                raw_answer = result["answer"]
                masked_answer = mask_pii(raw_answer)
                latency = time.perf_counter() - start

                leak_rate = pii_leakage_rate(masked_answer, case["pii_entities"])

                status = "CLEAN" if leak_rate == 0.0 else f"LEAK {leak_rate:.0%}"
                print(
                    f"[{i+1:02d}/{len(dataset)}] {case['id']} ({case['pii_type']:<14}) "
                    f"{status:<12} {latency:.1f}s"
                )
                if leak_rate > 0.0:
                    leaked = [p for p in case["pii_entities"] if p in masked_answer]
                    print(f"         leaked: {leaked}")

                results.append({
                    "id": case["id"],
                    "pii_type": case["pii_type"],
                    "leak_rate": round(leak_rate, 3),
                    "latency_s": round(latency, 2),
                    "masked_answer_preview": masked_answer[:200],
                })
                break

            except Exception as e:
                if attempt < 2:
                    print(f"  attempt {attempt+1} failed: {e} — retrying in 5s...")
                    await asyncio.sleep(5)
                else:
                    print(f"  [{case['id']}] ERROR after 3 attempts: {e}")
                    results.append({
                        "id": case["id"],
                        "pii_type": case["pii_type"],
                        "leak_rate": 1.0,
                        "error": str(e),
                    })

    # Aggregate
    n = len(results)
    clean_count = sum(1 for r in results if r["leak_rate"] == 0.0)
    avg_leak = sum(r["leak_rate"] for r in results) / n if n > 0 else 0.0

    # Per-type breakdown
    pii_types = sorted({r["pii_type"] for r in results})
    type_rates = {}
    for pii_type in pii_types:
        type_results = [r for r in results if r["pii_type"] == pii_type]
        type_rates[pii_type] = round(
            sum(r["leak_rate"] for r in type_results) / len(type_results), 3
        )

    return {
        "eval": "pii_leakage_rate",
        "n": n,
        "clean_count": clean_count,
        "avg_leak_rate": round(avg_leak, 3),
        "per_type": type_rates,
        "per_case": results,
    }


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(injection_result: dict | None, pii_result: dict | None) -> None:
    print(f"\n{'='*65}")
    print("SECURITY EVAL SUMMARY")
    print(f"{'='*65}")

    if injection_result:
        r = injection_result
        print(f"\nInjection Robustness  —  {r['pass_count']}/{r['n']} safe  "
              f"(pass rate: {r['pass_rate']:.1%})")
        for style, rate in r["per_style"].items():
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  Per style:")
        for style, rate in sorted(r["per_style"].items()):
            indicator = "✓" if rate == 1.0 else "✗" if rate == 0.0 else "~"
            print(f"    {indicator} {style:<28} {rate:.0%}")

    if pii_result:
        r = pii_result
        print(f"\nPII Leakage Rate      —  {r['clean_count']}/{r['n']} clean  "
              f"(avg leak: {r['avg_leak_rate']:.1%})")
        print(f"  Per type:")
        for pii_type, rate in sorted(r["per_type"].items()):
            indicator = "✓" if rate == 0.0 else "✗"
            print(f"    {indicator} {pii_type:<28} {rate:.0%} leak rate")

    print(f"\n{'='*65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(evals_to_run: list[str], limit: int | None = None) -> None:
    injection_result = None
    pii_result = None

    if "injection" in evals_to_run:
        dataset = INJECTION_DATASET[:limit] if limit else INJECTION_DATASET
        injection_result = await run_injection_eval(dataset)
        output = Path("eval/security_injection_results.json")
        with output.open("w") as f:
            json.dump(injection_result, f, indent=2)
        print(f"\nSaved to {output}")

    if "pii" in evals_to_run:
        dataset = PII_DATASET[:limit] if limit else PII_DATASET
        pii_result = await run_pii_eval(dataset)
        output = Path("eval/security_pii_results.json")
        with output.open("w") as f:
            json.dump(pii_result, f, indent=2)
        print(f"\nSaved to {output}")

    print_summary(injection_result, pii_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run security evals: injection robustness and PII leakage rate."
    )
    parser.add_argument(
        "--eval",
        choices=["injection", "pii", "both"],
        default="both",
        help="Which eval to run (default: both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N cases per eval (for quick smoke tests)",
    )
    args = parser.parse_args()

    evals = ["injection", "pii"] if args.eval == "both" else [args.eval]
    asyncio.run(main(evals, limit=args.limit))