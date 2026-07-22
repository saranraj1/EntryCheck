#!/usr/bin/env python3
"""
Stage 3 compatibility × determinism matrix runner (DR-006A §4, Step 7).

Usage:
    uv run python scripts/validation/stage3_determinism_matrix.py \
        --seeds 0 1 2 3 4 \
        --output-dir artifacts/pilot/stage3-finalization-v1

Produces:
    determinism-matrix.json — 4-cell matrix with repeatability + sensitivity results

Four cells: RF+KernelSHAP, RF+LIME, XGB+KernelSHAP, XGB+LIME.

Per DR-006A §9 definitions:
    Same-seed repeatability: same seed → same outputs
    Different-seed sensitivity: different seeds → outputs may differ (expected for stochastic)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from explaincheck.validation.determinism_matrix import run_cell

CELLS = [
    ("rf", "kernelshap"),
    ("rf", "lime"),
    ("xgb", "kernelshap"),
    ("xgb", "lime"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility × determinism matrix")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/pilot/stage3-finalization-v1")
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "determinism-matrix.json"
    if json_path.exists():
        print(f"[ERROR] {json_path} already exists. Will not overwrite.", file=sys.stderr)
        return 1

    print(f"Running determinism matrix — seeds: {args.seeds}")
    print(f"Output directory: {output_dir}")
    print()

    matrix: dict = {"cells": [], "seeds": args.seeds, "schema_version": "1.0"}
    all_schema_valid = True

    for model_name, explainer_name in CELLS:
        cell_id = f"{model_name}+{explainer_name}"
        print(f"  Cell: {cell_id}...", end=" ", flush=True)
        try:
            result = run_cell(model_name, explainer_name, args.seeds)
            rep = result["same_seed_repeatability"]
            status = "PASS" if rep["all_repeat_checks_pass"] else "WARN"
            print(
                f"{status} | same-seed attrs identical: {rep['same_attributions']} "
                f"(max diff: {rep['max_attribution_diff']:.2e}) "
                f"| schema valid: {result['schema_valid']}"
            )
            matrix["cells"].append(result)
            if not result["schema_valid"]:
                all_schema_valid = False
        except Exception as exc:
            print(f"ERROR: {exc}")
            matrix["cells"].append(
                {
                    "cell": cell_id,
                    "error": str(exc),
                    "schema_valid": False,
                }
            )
            all_schema_valid = False

    matrix["all_schema_valid"] = all_schema_valid
    matrix["note"] = (
        "Different outputs across seeds for stochastic methods (KernelSHAP, LIME) "
        "are expected and not classified as failures per DR-006A §9."
    )

    with json_path.open("w") as f:
        json.dump(matrix, f, indent=2, default=str)
    print(f"\nDeterminism matrix written: {json_path}")

    print()
    print("=== Determinism Matrix Summary ===")
    for cell in matrix["cells"]:
        cid = cell.get("cell", "unknown")
        if "error" in cell:
            print(f"  {cid}: ERROR — {cell['error']}")
            continue
        rep = cell["same_seed_repeatability"]
        print(
            f"  {cid}: repeat={'PASS' if rep['all_repeat_checks_pass'] else 'FAIL'} "
            f"| schema={'PASS' if cell['schema_valid'] else 'FAIL'}"
        )

    return 0 if all_schema_valid else 2


if __name__ == "__main__":
    sys.exit(main())
