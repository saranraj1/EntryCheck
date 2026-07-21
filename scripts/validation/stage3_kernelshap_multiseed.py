#!/usr/bin/env python3
"""
Stage 3 KernelSHAP multi-seed validation runner (DR-006A §4, Step 7).

Usage:
    uv run python scripts/validation/stage3_kernelshap_multiseed.py \
        --seeds 0 1 2 3 4 \
        --output-dir artifacts/pilot/stage3-finalization-v1

Produces:
    kernelshap-multiseed.csv         — per-seed metrics table
    kernelshap-multiseed-summary.json — 5-seed mean/SD/min/max + gate results

Gates applied to 5-seed mean (frozen, DR-006A §3):
    Mean cosine similarity >= 0.99
    Mean Spearman correlation >= 0.95
    Mean nonzero-feature sign agreement >= 0.95

Gate failures are recorded exactly as observed — never re-run or adjusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from explaincheck.validation.kernelshap_reference import run_kernelshap_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="KernelSHAP multi-seed validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pilot/stage3-finalization-v1"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "kernelshap-multiseed.csv"
    json_path = output_dir / "kernelshap-multiseed-summary.json"

    # Check: refuse to overwrite frozen artifacts unless explicitly allowed
    if csv_path.exists() or json_path.exists():
        print(
            f"[ERROR] Artifact files already exist in {output_dir}.\n"
            "To protect frozen artifacts, this script will not overwrite them.\n"
            "If this is a development re-run, move/rename existing files first.",
            file=sys.stderr,
        )
        return 1

    print(f"Running KernelSHAP multi-seed validation — seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    print()

    results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = run_kernelshap_seed(seed)
        results.append(r)
        status = "PASS" if r["all_gates_pass"] else "FAIL"
        print(
            f"{status} | cosine={r['mean_cosine']:.4f} "
            f"spearman={r['mean_spearman']:.4f} "
            f"sign={r['mean_sign_agree']:.4f} "
            f"t={r['runtime_ms']:.0f}ms"
        )

    # Write CSV
    fieldnames = [
        "seed", "mean_cosine", "mean_spearman", "mean_sign_agree",
        "mean_mae", "max_mae", "mean_topk_agree", "runtime_ms",
        "gate_cosine", "gate_spearman", "gate_sign_agree", "all_gates_pass",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-seed CSV written: {csv_path}")

    # Aggregate summary
    def _stats(key: str) -> dict:
        vals = [r[key] for r in results]
        return {
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    mean_cosine = statistics.mean(r["mean_cosine"] for r in results)
    mean_spearman = statistics.mean(r["mean_spearman"] for r in results)
    mean_sign_agree = statistics.mean(r["mean_sign_agree"] for r in results)

    # Frozen gates applied to 5-seed mean
    summary = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "frozen_gates": {
            "mean_cosine_threshold": 0.99,
            "mean_spearman_threshold": 0.95,
            "mean_sign_agree_threshold": 0.95,
        },
        "5_seed_mean_results": {
            "mean_cosine": mean_cosine,
            "mean_spearman": mean_spearman,
            "mean_sign_agree": mean_sign_agree,
        },
        "gate_results": {
            "cosine_pass": mean_cosine >= 0.99,
            "spearman_pass": mean_spearman >= 0.95,
            "sign_agree_pass": mean_sign_agree >= 0.95,
            "all_gates_pass": (mean_cosine >= 0.99 and mean_spearman >= 0.95 and mean_sign_agree >= 0.95),
        },
        "descriptive_stats": {
            "mean_cosine": _stats("mean_cosine"),
            "mean_spearman": _stats("mean_spearman"),
            "mean_sign_agree": _stats("mean_sign_agree"),
            "mean_mae": _stats("mean_mae"),
            "max_mae": _stats("max_mae"),
            "mean_topk_agree": _stats("mean_topk_agree"),
            "runtime_ms": _stats("runtime_ms"),
        },
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
    print(f"  Mean cosine:       {mean_cosine:.4f}  >= 0.99  -> {'PASS' if mean_cosine >= 0.99 else 'FAIL'}")
    print(f"  Mean Spearman:     {mean_spearman:.4f}  >= 0.95  -> {'PASS' if mean_spearman >= 0.95 else 'FAIL'}")
    print(f"  Mean sign agree:   {mean_sign_agree:.4f}  >= 0.95  -> {'PASS' if mean_sign_agree >= 0.95 else 'FAIL'}")
    all_pass = summary["gate_results"]["all_gates_pass"]
    print(f"\n  Overall: {'ALL GATES PASS (PASS)' if all_pass else 'ONE OR MORE GATES FAIL (FAIL)'}")

    return 0 if all_pass else 2  # exit 2 = gates failed (not a script error)


if __name__ == "__main__":
    sys.exit(main())
