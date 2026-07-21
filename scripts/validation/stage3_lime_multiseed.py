#!/usr/bin/env python3
"""
Stage 3 LIME multi-seed validation runner (DR-006A §4, Step 7).

Usage:
    uv run python scripts/validation/stage3_lime_multiseed.py \
        --seeds 0 1 2 3 4 \
        --output-dir artifacts/pilot/stage3-finalization-v1

Produces:
    lime-multiseed.csv              — per-seed metrics table
    lime-multiseed-summary.json     — 5-seed mean/SD/min/max + gate results

Frozen gates (applied to 5-seed mean, DR-006A §3):
    Mean cosine similarity >= 0.95
    Mean nonzero-feature sign agreement >= 0.90
    Mean Top-k signal recall >= 0.90 (k=3)

Descriptive (no frozen gate):
    Spearman rank correlation
    Dense attribution variation across seeds
    Actual numeric kernel width
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from explaincheck.validation.lime_reference import run_lime_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="LIME multi-seed validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pilot/stage3-finalization-v1"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "lime-multiseed.csv"
    json_path = output_dir / "lime-multiseed-summary.json"

    if csv_path.exists() or json_path.exists():
        print(
            f"[ERROR] Artifact files already exist in {output_dir}.\n"
            "To protect frozen artifacts, this script will not overwrite them.",
            file=sys.stderr,
        )
        return 1

    print(f"Running LIME multi-seed validation — seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    print()

    results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = run_lime_seed(seed)
        results.append(r)
        status = "PASS" if r["all_gates_pass"] else "FAIL"
        print(
            f"{status} | cosine={r['mean_cosine']:.4f} "
            f"sign={r['mean_sign_agree']:.4f} "
            f"topk={r['mean_topk_recall']:.4f} "
            f"spearman={r['mean_spearman']:.4f} "
            f"t={r['runtime_ms']:.0f}ms"
        )

    fieldnames = [
        "seed", "mean_cosine", "mean_sign_agree", "mean_topk_recall",
        "mean_spearman", "std_attribution", "kernel_width", "runtime_ms",
        "gate_cosine", "gate_sign_agree", "gate_topk_recall", "all_gates_pass",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-seed CSV written: {csv_path}")

    def _stats(key: str) -> dict:
        vals = [r[key] for r in results]
        return {
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    mean_cosine = statistics.mean(r["mean_cosine"] for r in results)
    mean_sign_agree = statistics.mean(r["mean_sign_agree"] for r in results)
    mean_topk_recall = statistics.mean(r["mean_topk_recall"] for r in results)
    mean_spearman = statistics.mean(r["mean_spearman"] for r in results)
    kernel_width = results[0]["kernel_width"]  # same for all seeds

    summary = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "frozen_gates": {
            "mean_cosine_threshold": 0.95,
            "mean_sign_agree_threshold": 0.90,
            "mean_topk_recall_threshold": 0.90,
        },
        "5_seed_mean_results": {
            "mean_cosine": mean_cosine,
            "mean_sign_agree": mean_sign_agree,
            "mean_topk_recall": mean_topk_recall,
        },
        "gate_results": {
            "cosine_pass": mean_cosine >= 0.95,
            "sign_agree_pass": mean_sign_agree >= 0.90,
            "topk_recall_pass": mean_topk_recall >= 0.90,
            "all_gates_pass": (mean_cosine >= 0.95 and mean_sign_agree >= 0.90 and mean_topk_recall >= 0.90),
        },
        "descriptive_stats": {
            "mean_cosine": _stats("mean_cosine"),
            "mean_sign_agree": _stats("mean_sign_agree"),
            "mean_topk_recall": _stats("mean_topk_recall"),
            "mean_spearman": _stats("mean_spearman"),
            "std_attribution": _stats("std_attribution"),
            "runtime_ms": _stats("runtime_ms"),
        },
        "kernel_width_actual": kernel_width,
        "per_seed": results,
        "note": (
            "Gate failures recorded exactly as observed — seeds never removed "
            "or settings adjusted after inspecting outcomes (DR-006A §3)."
        ),
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON written:  {json_path}")

    print()
    print("=== Gate Summary (5-seed mean) ===")
    print(f"  Mean cosine:       {mean_cosine:.4f}  >= 0.95  -> {'PASS' if mean_cosine >= 0.95 else 'FAIL'}")
    print(f"  Mean sign agree:   {mean_sign_agree:.4f}  >= 0.90  -> {'PASS' if mean_sign_agree >= 0.90 else 'FAIL'}")
    print(f"  Mean top-k recall: {mean_topk_recall:.4f}  >= 0.90  -> {'PASS' if mean_topk_recall >= 0.90 else 'FAIL'}")
    print(f"  Descriptive Spearman: {mean_spearman:.4f}  (no gate)")
    print(f"  Kernel width (actual): {kernel_width:.4f}")
    all_pass = summary["gate_results"]["all_gates_pass"]
    print(f"\n  Overall: {'ALL GATES PASS (PASS)' if all_pass else 'ONE OR MORE GATES FAIL (FAIL)'}")

    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
