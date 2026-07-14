"""
ExplainCheck — Synthetic linear-logit pilot runner.

Migrates Phase 0 end-to-end pipeline into the package.
Writes all required outputs to the Stage 2 artifact directory.

Output contract (DR-002):
    run-manifest.json, environment.json, benchmark.json,
    raw-results.parquet, tidy-results.csv, model-performance.csv,
    failures.csv, manual-metric-validation.json, model-card.md,
    SHA256SUMS.txt, tables/, figures/, paper-snippets/
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explaincheck import __protocol_version__, __study_id__, __version__
from explaincheck.contracts import (
    AttributionRecord,
    ExplainerName,
    FailureRecord,
    MetricResult,
    ModelFamily,
    RunLabel,
    RunStatus,
)
from explaincheck.datasets.synthetic import (
    BETA_TRUE,
    FEATURE_NAMES,
    dataset_record,
    generate,
    split,
    split_record,
)
from explaincheck.explainers.exact_linear import ExactLinearExplainer, RandomizedNegativeControl
from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single
from explaincheck.metrics.stability.top_k_jaccard import TopKJaccardStability, jaccard
from explaincheck.models.logistic_regression import LogisticRegressionAdapter
from explaincheck.provenance import (
    hash_file,
    new_run_id,
    register_artifacts,
    snapshot_environment,
    utc_now_iso,
)

# Frozen from DR-001-PHASE1
FROZEN_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
EVAL_SIZES = [50, 100, 200]
KMAX = 3
SIGMA = 0.05

# Phase 0 reproduction tolerances (DR-002)
TOLERANCES = {
    "roc_auc_mean": (0.878, 0.005),
    "coef_cosine_mean": (0.998, 0.002),
    "exact_fidelity_n200": (1.764, 0.020),
    "neg_fidelity_n200": (0.753, 0.030),
    "exact_stability_n200": (0.957, 0.015),
    "neg_stability_n200": (0.259, 0.020),
}


# ---------------------------------------------------------------------------
# Mann-Whitney AUC (migrated from Phase 0)
# ---------------------------------------------------------------------------

def _auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bootstrap_ci(values: np.ndarray, seed: int, reps: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(reps)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Manual validation fixture (must match Phase 0 exactly)
# ---------------------------------------------------------------------------

def manual_validation() -> dict[str, Any]:
    """
    Reproduce the Phase 0 hand-computation fixture.

    Expected: fidelity AOPC@2 = 2.25, stability Jaccard@2 = 1.0.
    These must pass exactly (no tolerance).
    """
    w = np.array([1.0, 0.5, 0.2])
    b = 0.0
    baseline = np.zeros(3)
    x = np.array([2.0, 1.0, 1.0])
    xp = np.array([2.01, 0.98, 1.02])
    a = (x - baseline) * w
    ap = (xp - baseline) * w

    fidelity = deletion_fidelity_aopc_single(x, a, w, b, baseline, kmax=2)
    stability = jaccard(a, ap, k=2)

    assert np.allclose(a, [2.0, 0.5, 0.2]), f"Attribution mismatch: {a}"
    assert abs(fidelity - 2.25) < 1e-12, f"Fidelity fixture failed: got {fidelity}"
    assert abs(stability - 1.0) < 1e-12, f"Stability fixture failed: got {stability}"

    return {
        "fixture": {"x": x.tolist(), "x_prime": xp.tolist(), "weights": w.tolist(), "baseline": baseline.tolist()},
        "attribution": a.tolist(),
        "fidelity_aopc_at_2_hand_expected": 2.25,
        "fidelity_aopc_at_2_computed": fidelity,
        "stability_jaccard_at_2_hand_expected": 1.0,
        "stability_jaccard_at_2_computed": stability,
        "status": "passed",
    }


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def _run_seed(seed: int, run_id: str) -> tuple[list[dict], dict, list[dict]]:
    """Run one seed. Returns (metric_rows, model_row, failure_rows)."""
    X, y = generate(seed)
    X_tr, X_te, y_tr, y_te = split(X, y, seed)
    baseline = X_tr.mean(axis=0)

    # Fit model
    lr = LogisticRegressionAdapter()
    model_record = lr.fit(X_tr, y_tr, seed=seed)

    # Model performance
    probs = lr.predict_proba(X_te)[:, 1]
    auc = _auc_rank(y_te, probs)
    acc = float(np.mean(lr.predict(X_te) == y_te))
    coef_cos = float(np.dot(lr.weights, BETA_TRUE) / (np.linalg.norm(lr.weights) * np.linalg.norm(BETA_TRUE)))

    model_row = {
        "seed": seed, "accuracy": acc, "roc_auc": auc,
        "coefficient_cosine_to_truth": coef_cos, "fit_ms": model_record.fit_ms,
    }

    # Explainers
    exact_exp = ExactLinearExplainer()
    exact_exp.fit(lr, X_tr, FEATURE_NAMES, seed=seed)
    neg_exp = RandomizedNegativeControl()
    neg_exp.fit(lr, X_tr, FEATURE_NAMES, seed=seed)

    # Evaluation sub-samples
    rng_eval = np.random.default_rng(seed + 5000)
    order = rng_eval.permutation(len(X_te))

    metric_rows: list[dict] = []
    failure_rows: list[dict] = []
    split_rec = split_record(X_tr, X_te, y_tr, y_te, seed)

    for n_eval in EVAL_SIZES:
        Xe = X_te[order[:n_eval]]
        sample_ids = [f"s{seed}-n{n_eval}-{i}" for i in range(n_eval)]

        for method_name, exp in [("exact_linear", exact_exp), ("randomized_negative_control", neg_exp)]:
            attr_records = exp.explain(
                Xe,
                run_id=run_id,
                dataset="synthetic_linear",
                seed=seed,
                model_family=ModelFamily.LOGISTIC_REGRESSION,
                model_hash=model_record.model_hash,
                sample_ids=sample_ids,
                protocol_version=__protocol_version__,
                model=lr,
            )

            good_attrs = [r for r in attr_records if isinstance(r, AttributionRecord)]
            fails = [r for r in attr_records if isinstance(r, FailureRecord)]
            for f in fails:
                failure_rows.append(f.model_dump())

            if not good_attrs:
                continue

            # Fidelity
            t_f = time.perf_counter()
            fid_vals = np.array([
                deletion_fidelity_aopc_single(
                    Xe[i], np.array(r.attribution), lr.weights, lr.bias, baseline, KMAX
                )
                for i, r in enumerate(good_attrs)
            ])
            f_lo, f_hi = _bootstrap_ci(fid_vals, seed + n_eval)
            rt_f = (time.perf_counter() - t_f) * 1000

            # Stability (prediction-conditioned)
            t_s = time.perf_counter()
            rng_pert = np.random.default_rng(seed + 9000)
            Xep = Xe + rng_pert.normal(0, SIGMA, size=Xe.shape)
            pred_orig = (_sigmoid(Xe @ lr.weights + lr.bias) >= 0.5).astype(int)
            pred_pert = (_sigmoid(Xep @ lr.weights + lr.bias) >= 0.5).astype(int)
            preserved = pred_orig == pred_pert
            n_rejected = int((~preserved).sum())

            stab_vals_list = []
            for i, r in enumerate(good_attrs):
                if not preserved[i]:
                    continue
                attr_orig = np.array(r.attribution)
                if method_name == "exact_linear":
                    attr_pert = (Xep[i] - baseline) * lr.weights
                else:
                    exact_pert = (Xep[i] - baseline) * lr.weights
                    rng_neg = np.random.default_rng(seed + 1)
                    attr_pert = exact_pert[rng_neg.permutation(exact_pert.size)]
                stab_vals_list.append(jaccard(attr_orig, attr_pert, KMAX))

            stab_vals = np.array(stab_vals_list) if stab_vals_list else np.array([float("nan")])
            s_lo, s_hi = _bootstrap_ci(stab_vals[~np.isnan(stab_vals)], seed + n_eval + 1) \
                if not np.all(np.isnan(stab_vals)) else (float("nan"), float("nan"))
            rt_s = (time.perf_counter() - t_s) * 1000
            preserved_rate = float(preserved.mean())

            metric_rows.extend([
                {
                    "seed": seed, "sample_size": n_eval, "explainer": method_name,
                    "metric": "deletion_fidelity_aopc_at_3",
                    "estimate": float(np.mean(fid_vals)), "ci_low": f_lo, "ci_high": f_hi,
                    "n": len(fid_vals), "prediction_preservation_rate": preserved_rate,
                    "n_rejected": n_rejected, "runtime_ms": rt_f,
                },
                {
                    "seed": seed, "sample_size": n_eval, "explainer": method_name,
                    "metric": "stability_top3_jaccard",
                    "estimate": float(np.mean(stab_vals[~np.isnan(stab_vals)])) if not np.all(np.isnan(stab_vals)) else float("nan"),
                    "ci_low": s_lo, "ci_high": s_hi,
                    "n": len(stab_vals_list), "prediction_preservation_rate": preserved_rate,
                    "n_rejected": n_rejected, "runtime_ms": rt_s,
                },
            ])

    return metric_rows, model_row, failure_rows


# ---------------------------------------------------------------------------
# Full pilot run
# ---------------------------------------------------------------------------

def run_pilot(
    seeds: list[int] = FROZEN_SEEDS,
    eval_sizes: list[int] = EVAL_SIZES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run the full synthetic pilot across all seeds.

    Returns (results_df, models_df, failures_df, timing).
    """
    run_id = new_run_id()
    all_metrics: list[dict] = []
    all_models: list[dict] = []
    all_failures: list[dict] = []
    t_start = time.perf_counter()

    for seed in seeds:
        m_rows, model_row, f_rows = _run_seed(seed, run_id)
        all_metrics.extend(m_rows)
        all_models.append(model_row)
        all_failures.extend(f_rows)

    elapsed = time.perf_counter() - t_start
    return (
        pd.DataFrame(all_metrics),
        pd.DataFrame(all_models),
        pd.DataFrame(all_failures) if all_failures else pd.DataFrame(),
        {"run_id": run_id, "elapsed_seconds": elapsed},
    )


# ---------------------------------------------------------------------------
# Reproduction check
# ---------------------------------------------------------------------------

def check_reproduction(results: pd.DataFrame, models: pd.DataFrame) -> dict[str, Any]:
    """
    Compare Stage 2 results against Phase 0 reference values within declared tolerances.
    Returns a dict with pass/fail status per metric.
    """
    n200 = results[results.sample_size == 200].groupby(["explainer", "metric"])["estimate"].mean()

    checks: dict[str, Any] = {}

    def chk(key: str, got: float, ref: float, tol: float) -> dict:
        ok = abs(got - ref) <= tol
        return {"got": round(got, 6), "reference": ref, "tolerance": tol, "passed": ok, "delta": round(got - ref, 6)}

    checks["roc_auc_mean"] = chk("roc_auc_mean", float(models.roc_auc.mean()), 0.878, 0.005)
    checks["coef_cosine_mean"] = chk("coef_cosine_mean", float(models.coefficient_cosine_to_truth.mean()), 0.998, 0.002)

    try:
        checks["exact_fidelity_n200"] = chk(
            "exact_fidelity_n200",
            float(n200.loc[("exact_linear", "deletion_fidelity_aopc_at_3")]), 1.764, 0.020
        )
        checks["neg_fidelity_n200"] = chk(
            "neg_fidelity_n200",
            float(n200.loc[("randomized_negative_control", "deletion_fidelity_aopc_at_3")]), 0.753, 0.030
        )
        checks["exact_stability_n200"] = chk(
            "exact_stability_n200",
            float(n200.loc[("exact_linear", "stability_top3_jaccard")]), 0.957, 0.015
        )
        checks["neg_stability_n200"] = chk(
            "neg_stability_n200",
            float(n200.loc[("randomized_negative_control", "stability_top3_jaccard")]), 0.259, 0.020
        )
    except KeyError as e:
        checks["_error"] = f"Missing metric key: {e}"

    all_passed = all(v.get("passed", False) for v in checks.values() if isinstance(v, dict))
    checks["_all_passed"] = all_passed
    return checks


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    results: pd.DataFrame,
    models: pd.DataFrame,
    failures: pd.DataFrame,
    validation: dict[str, Any],
    timing: dict[str, Any],
    reproduction: dict[str, Any],
    out_dir: Path,
    config_hash: str = "unknown",
) -> str:
    """Write all required Stage 2 artifacts. Returns the run_id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    paper_dir = out_dir / "paper-snippets"
    for d in [tables_dir, figures_dir, paper_dir]:
        d.mkdir(exist_ok=True)

    run_id = timing["run_id"]

    # Environment
    env = snapshot_environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    # Raw results as parquet
    results.to_parquet(out_dir / "raw-results.parquet", index=False)

    # Tidy CSV
    results.to_csv(out_dir / "tidy-results.csv", index=False)

    # Model performance
    models.to_csv(out_dir / "model-performance.csv", index=False)

    # Failures
    if failures is not None and len(failures):
        failures.to_csv(out_dir / "failures.csv", index=False)
    else:
        pd.DataFrame(columns=["run_id", "timestamp", "dataset", "seed", "failure_reason"]).to_csv(
            out_dir / "failures.csv", index=False
        )

    # Manual validation fixture
    (out_dir / "manual-metric-validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    # Summary tables
    summary = results.groupby(["sample_size", "explainer", "metric"])["estimate"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    summary.to_csv(tables_dir / "table-s2-pilot-summary.csv", index=False)

    variance = results.groupby(["sample_size", "explainer", "metric"])["estimate"].std().reset_index(
        name="sd_across_seeds"
    )
    variance.to_csv(tables_dir / "table-s2-variance.csv", index=False)

    # LaTeX table
    final = summary[summary.sample_size == 200].copy()
    lines = [
        "\\begin{tabular}{llrr}", "\\toprule",
        "Explainer & Metric & Mean & SD \\\\", "\\midrule"
    ]
    for _, r in final.iterrows():
        e = str(r.explainer).replace("_", "\\_")
        m = str(r.metric).replace("_", "\\_")
        lines.append(f"{e} & {m} & {r['mean']:.3f} & {r['std']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables_dir / "table-s2-pilot-summary.tex").write_text("\n".join(lines), encoding="utf-8")

    # Figures
    _write_figures(results, figures_dir)

    # Benchmark JSON
    model_summary = {
        c: {"mean": float(models[c].mean()), "sd": float(models[c].std())}
        for c in ["accuracy", "roc_auc", "coefficient_cosine_to_truth", "fit_ms"]
    }
    n200 = final.pivot(index="explainer", columns="metric", values="mean")
    result_records = []
    for _, r in final.iterrows():
        result_records.append({
            "explainer": r.explainer, "metric": r.metric,
            "sampleSize": int(r.sample_size), "estimate": float(r["mean"]),
            "sdAcrossSeeds": float(r["std"]),
        })

    benchmark = {
        "schemaVersion": "1.0.0",
        "protocolVersion": __protocol_version__,
        "studyId": __study_id__,
        "runId": run_id,
        "runLabel": "infrastructure-pilot",
        "status": "pilot-not-confirmatory",
        "design": {
            "seeds": FROZEN_SEEDS,
            "evaluationSizes": EVAL_SIZES,
            "features": FEATURE_NAMES,
            "trueCoefficients": BETA_TRUE.tolist(),
            "intercept": -0.15,
            "nGeneratedPerSeed": 3000,
            "explainers": ["exact_linear", "randomized_negative_control"],
            "metrics": ["deletion_fidelity_aopc_at_3", "stability_top3_jaccard"],
            "kmax": KMAX,
            "sigma": SIGMA,
        },
        "manualValidation": validation,
        "modelPerformance": model_summary,
        "reproductionCheck": reproduction,
        "resultsAtN200": result_records,
        "timing": timing,
        "limitations": [
            "Pilot uses a controlled linear-logit setting; does not establish performance on nonlinear or real-world data.",
            "Randomized attribution is a negative control, not a competitive explainer.",
            "Pilot thresholds are engineering diagnostics and are not universal XAI standards.",
            "Stage 2 infrastructure-pilot results must not be interpreted as evidence about SHAP, LIME, XGBoost, or real-world datasets.",
        ],
    }
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    # Model card
    model_card = f"""# Model Card — Synthetic Linear Logistic Pilot (Stage 2)

## Status
Infrastructure-pilot artifact (Stage 2). Not confirmatory. Not suitable for deployment or substantive decision-making.

## Run ID
`{run_id}`

## Model details
- Model: custom L2-regularized logistic regression fitted by deterministic full-batch gradient descent
- Data: synthetic independent Gaussian features
- True coefficients: `{BETA_TRUE.tolist()}`
- Intercept: -0.15
- Seeds: `{FROZEN_SEEDS}`

## Intended use
Validate ExplainCheck Stage 2 package contracts, metric implementations, provenance, and report generation. Migrated from Phase 0.

## Predictive performance across seeds
- Accuracy: {models.accuracy.mean():.3f} ± {models.accuracy.std():.3f}
- ROC AUC: {models.roc_auc.mean():.3f} ± {models.roc_auc.std():.3f}
- Coefficient cosine similarity to generating coefficients: {models.coefficient_cosine_to_truth.mean():.3f} ± {models.coefficient_cosine_to_truth.std():.3f}

## Explanation methods
- Exact linear attribution: learned coefficient × deviation from training mean (control reference)
- Randomized negative control: within-sample permutation of exact attribution values (negative control)

## Metrics
- Deletion fidelity AOPC@3: mean cumulative absolute logit change after masking top-ranked features to training mean
- Stability Top-3 Jaccard: overlap of top features after small Gaussian perturbation (σ=0.05), restricted to prediction-preserving pairs

## Phase 0 reproduction
{json.dumps(reproduction, indent=2)}

## Limitations
This artifact validates Stage 2 machinery. It is not evidence that ExplainCheck outperforms Quantus or OpenXAI, and is not a confirmatory result.
"""
    (out_dir / "model-card.md").write_text(model_card, encoding="utf-8")

    # Paper snippets
    _write_paper_snippets(results, models, n200, run_id, paper_dir)

    # SHA256SUMS
    checksums = register_artifacts(out_dir)
    checksum_lines = [f"{v['sha256']}  {k}" for k, v in sorted(checksums.items())]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines), encoding="utf-8")

    # Run manifest
    checksums_with_manifest = register_artifacts(out_dir)
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "studyId": __study_id__,
        "protocolVersion": __protocol_version__,
        "runLabel": "infrastructure-pilot",
        "status": "pilot-not-confirmatory",
        "configHash": config_hash,
        "createdAt": utc_now_iso(),
        "environment": env,
        "seeds": FROZEN_SEEDS,
        "datasets": ["synthetic_linear"],
        "models": ["logistic_regression"],
        "explainers": ["exact_linear", "randomized_negative_control"],
        "files": checksums_with_manifest,
        "nSuccesses": int((results.estimate.notna()).sum()),
        "nFailures": len(failures) if failures is not None else 0,
        "nExcluded": 0,
        "elapsedSeconds": timing["elapsed_seconds"],
        "limitations": benchmark["limitations"],
        "osfRegistrationUrl": None,
    }
    (out_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return run_id


def _write_figures(results: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    colors = {"exact_linear": "#2783DE", "randomized_negative_control": "#D5803B"}
    labels = {"exact_linear": "Exact linear (reference)", "randomized_negative_control": "Randomized (negative control)"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, metric, title, ylabel in [
        (axes[0], "deletion_fidelity_aopc_at_3", "Fidelity under top-feature deletion", "Mean absolute logit change"),
        (axes[1], "stability_top3_jaccard", "Stability (prediction-preserving noise)", "Top-3 Jaccard similarity"),
    ]:
        sub = results[(results.metric == metric) & (results.sample_size == 200)]
        for xi, method in enumerate(["exact_linear", "randomized_negative_control"]):
            vals = sub[sub.explainer == method]["estimate"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            lo, hi = np.quantile(vals, [0.025, 0.975])
            ax.errorbar(xi, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                        color=colors[method], markersize=8, capsize=5, linewidth=2, label=labels[method])
        ax.set_xticks([0, 1], ["Exact\nreference", "Negative\ncontrol"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E6E5E3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#F9F8F7")

    fig.suptitle("ExplainCheck Stage 2 synthetic pilot · 10 seeds · n=200", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color="#2C2C2B")
    fig.savefig(figures_dir / "fig-s2-pilot-metrics.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig-s2-pilot-metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Variance plot
    var = results.groupby(["sample_size", "explainer", "metric"])["estimate"].std().reset_index(name="sd_across_seeds")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    markers = {"deletion_fidelity_aopc_at_3": "o", "stability_top3_jaccard": "s"}
    for (method, metric), g in var.groupby(["explainer", "metric"]):
        ax.plot(g.sample_size, g.sd_across_seeds, marker=markers.get(metric, "o"),
                color=colors.get(method, "gray"), linestyle="-" if "fidelity" in metric else "--",
                linewidth=2, markersize=6,
                label=f"{labels.get(method, method)} · {'fidelity' if 'fidelity' in metric else 'stability'}")
    ax.set_title("Variance decreases as evaluation sample size increases", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Explanations evaluated per seed")
    ax.set_ylabel("Standard deviation across seeds")
    ax.set_facecolor("#F9F8F7")
    ax.grid(color="#E6E5E3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(figures_dir / "fig-s2-variance.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig-s2-variance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_paper_snippets(
    results: pd.DataFrame,
    models: pd.DataFrame,
    n200: Any,
    run_id: str,
    paper_dir: Path,
) -> None:
    methods_text = f"""# Generated Methods Snippet — Stage 2 Synthetic Pilot

For each of ten prespecified random seeds, 3,000 observations with eight independent standard-normal features were generated. Binary outcomes were sampled from a logistic model with coefficients (1.50, −1.20, 0.90, −0.70, 0, 0, 0, 0) and intercept −0.15. Data were divided into 80% training and 20% test partitions using a seed-specific deterministic permutation. An L2-regularized logistic-regression model was fitted by full-batch gradient descent (steps=1800, lr=0.12, λ=10⁻³). Local exact-linear attributions were defined as the learned coefficient multiplied by the feature's deviation from the training-set mean. A within-instance randomized permutation of these attributions served as a negative control.

Deletion fidelity was operationalized as the mean cumulative absolute change in model logit after replacing the three highest-ranked features with their training-set means. Stability was operationalized as Top-3 Jaccard overlap after Gaussian input perturbation (σ=0.05), restricted to pairs whose predicted class was preserved. Metrics were estimated at evaluation sample sizes of 50, 100, and 200 per seed. The pilot reports across-seed variation and 1,000-replicate within-run bootstrap intervals; it is not a confirmatory hypothesis test.

Run ID: {run_id}. Label: infrastructure-pilot. Must not be interpreted as evidence about SHAP, LIME, XGBoost, or real-world datasets.
"""
    (paper_dir / "methods-generated.md").write_text(methods_text, encoding="utf-8")

    try:
        ef = float(n200.loc["exact_linear", "deletion_fidelity_aopc_at_3"])
        nf = float(n200.loc["randomized_negative_control", "deletion_fidelity_aopc_at_3"])
        es = float(n200.loc["exact_linear", "stability_top3_jaccard"])
        ns = float(n200.loc["randomized_negative_control", "stability_top3_jaccard"])
    except (KeyError, TypeError):
        ef = nf = es = ns = float("nan")

    results_text = f"""# Pilot Results Snippet — Stage 2

Across ten seeds, the fitted logistic model achieved mean ROC AUC {models.roc_auc.mean():.3f} (SD {models.roc_auc.std():.3f}) and recovered the generating coefficient direction with cosine similarity {models.coefficient_cosine_to_truth.mean():.3f} (SD {models.coefficient_cosine_to_truth.std():.3f}). At 200 evaluated explanations per seed, exact linear attributions achieved mean deletion fidelity {ef:.3f}, compared with {nf:.3f} for the randomized negative control. Mean Top-3 stability was {es:.3f} for exact attributions and {ns:.3f} for the negative control. These values validate the expected directional behavior of the Stage 2 metrics but must not be interpreted as evidence about SHAP, LIME, or real-world deployments.

Run ID: {run_id}. Label: infrastructure-pilot.
"""
    (paper_dir / "results-generated.md").write_text(results_text, encoding="utf-8")
