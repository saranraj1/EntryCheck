"""
Stage 2 golden tests — frozen hand fixtures and Phase 0 regression checks.

These tests verify that the implementation produces exactly the values
declared in DR-002-PHASE1-STAGE2 and matches Phase 0 reference outputs
within tolerance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hand fixture validation (must pass EXACTLY — no tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_hand_fixture_fidelity_aopc2_exact() -> None:
    """DR-002 requirement: Fidelity AOPC@2 = 2.25 exactly."""
    from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single

    w = np.array([1.0, 0.5, 0.2])
    b = 0.0
    baseline = np.zeros(3)
    x = np.array([2.0, 1.0, 1.0])
    a = (x - baseline) * w

    result = deletion_fidelity_aopc_single(x, a, w, b, baseline, kmax=2)

    assert result == 2.25, f"FIDELITY FIXTURE FAILED: expected exactly 2.25, got {result!r}"


@pytest.mark.golden
def test_hand_fixture_stability_jaccard2_exact() -> None:
    """DR-002 requirement: Stability Jaccard@2 = 1.0 exactly."""
    from explaincheck.metrics.stability.top_k_jaccard import jaccard

    a = np.array([2.0, 0.5, 0.2])
    ap = np.array([2.01, 0.49, 0.21])

    result = jaccard(a, ap, k=2)

    assert result == 1.0, f"STABILITY FIXTURE FAILED: expected exactly 1.0, got {result!r}"


@pytest.mark.golden
def test_manual_validation_runner() -> None:
    """The pilot runner's manual_validation() must return status='passed'."""
    from explaincheck.pilot.runner import manual_validation

    v = manual_validation()
    assert v["status"] == "passed"
    assert v["fidelity_aopc_at_2_computed"] == 2.25
    assert v["stability_jaccard_at_2_computed"] == 1.0


# ---------------------------------------------------------------------------
# Phase 0 regression tests — must reproduce within declared tolerance
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.slow
def test_phase0_reproduction_all_seeds() -> None:
    """
    Reproduce Phase 0 results within DR-002 tolerance on all 10 seeds.

    If any value is outside tolerance, the test fails and reports the deviation.
    Per DR-002: do NOT adjust tolerance or implementation to force a pass.
    """
    from explaincheck.pilot.runner import check_reproduction, run_pilot

    results, models, failures, timing = run_pilot()
    checks = check_reproduction(results, models)

    failures_found = {
        k: v for k, v in checks.items() if isinstance(v, dict) and not v.get("passed")
    }

    report_lines = ["\nPhase 0 Reproduction Report:"]
    for k, v in checks.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            status = "PASS" if v["passed"] else "FAIL"
            report_lines.append(
                f"  {status}: {k}  got={v['got']}  ref={v['reference']}  "
                f"delta={v['delta']:+.6f}  tol=±{v['tolerance']}"
            )

    if failures_found:
        import warnings

        warnings.warn("\n".join(report_lines))
        assert False, (
            f"Phase 0 reproduction FAILED for: {list(failures_found.keys())}\n"
            + "\n".join(report_lines)
            + "\n\nPer DR-002: do NOT adjust tolerances. Produce a variance report."
        )
    else:
        assert True


# ---------------------------------------------------------------------------
# Artifact schema test
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.slow
def test_artifacts_schema_valid(tmp_path) -> None:
    """All output files must be present and the run-manifest.json must parse."""
    from explaincheck.pilot.runner import (
        check_reproduction,
        manual_validation,
        run_pilot,
        write_outputs,
    )

    results, models, failures, timing = run_pilot(seeds=[11, 23])
    reproduction = check_reproduction(results, models)
    validation = manual_validation()
    out_dir = tmp_path / "stage2-test"
    run_id = write_outputs(results, models, failures, validation, timing, reproduction, out_dir)

    required_files = [
        "run-manifest.json",
        "environment.json",
        "benchmark.json",
        "raw-results.parquet",
        "tidy-results.csv",
        "model-performance.csv",
        "failures.csv",
        "manual-metric-validation.json",
        "model-card.md",
        "SHA256SUMS.txt",
        "tables/table-s2-pilot-summary.csv",
        "tables/table-s2-variance.csv",
        "tables/table-s2-pilot-summary.tex",
        "figures/fig-s2-pilot-metrics.pdf",
        "figures/fig-s2-pilot-metrics.png",
        "figures/fig-s2-variance.pdf",
        "figures/fig-s2-variance.png",
        "paper-snippets/methods-generated.md",
        "paper-snippets/results-generated.md",
    ]
    missing = [f for f in required_files if not (out_dir / f).exists()]
    assert not missing, f"Missing required output files: {missing}"

    manifest = json.loads((out_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["runLabel"] == "infrastructure-pilot"
    assert manifest["osfRegistrationUrl"] is None
    assert "files" in manifest
    assert len(manifest["files"]) > 0

    validation_json = json.loads(
        (out_dir / "manual-metric-validation.json").read_text(encoding="utf-8")
    )
    assert validation_json["status"] == "passed"
    assert validation_json["fidelity_aopc_at_2_computed"] == 2.25
    assert validation_json["stability_jaccard_at_2_computed"] == 1.0
