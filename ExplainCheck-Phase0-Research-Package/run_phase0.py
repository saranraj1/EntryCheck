from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/data/explaincheck_phase0')
ART = ROOT / 'artifacts' / 'pilot-v1'
TABLES = ART / 'tables'
FIGURES = ART / 'figures'
PAPER = ART / 'paper-snippets'
for p in [ART, TABLES, FIGURES, PAPER]:
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
SAMPLE_SIZES = [50, 100, 200]
BETA_TRUE = np.array([1.50, -1.20, 0.90, -0.70, 0.0, 0.0, 0.0, 0.0])
FEATURES = [f'x{i+1}' for i in range(len(BETA_TRUE))]


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, *, steps: int = 1800, lr: float = 0.12, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def generate(seed: int, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(BETA_TRUE)))
    logits = X @ BETA_TRUE - 0.15
    y = rng.binomial(1, sigmoid(logits))
    return X, y


def split(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def exact_attributions(X: np.ndarray, w: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (X - baseline) * w


def randomized_negative_control(A: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty_like(A)
    for i, row in enumerate(A):
        out[i] = row[rng.permutation(row.size)]
    return out


def topk_idx(row: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(row), kind='stable')[:k]


def deletion_fidelity(X: np.ndarray, A: np.ndarray, w: np.ndarray, b: float, baseline: np.ndarray, kmax: int = 3) -> np.ndarray:
    base_logits = X @ w + b
    scores = np.zeros(len(X))
    for i in range(len(X)):
        order = topk_idx(A[i], kmax)
        drops = []
        xm = X[i].copy()
        for j in order:
            xm[j] = baseline[j]
            drops.append(abs(base_logits[i] - (xm @ w + b)))
        scores[i] = float(np.mean(drops))
    return scores


def jaccard(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    sa, sb = set(topk_idx(a, k).tolist()), set(topk_idx(b, k).tolist())
    return len(sa & sb) / len(sa | sb)


def stability_scores(X: np.ndarray, A: np.ndarray, w: np.ndarray, b: float, baseline: np.ndarray, explainer: str, seed: int, sigma: float = 0.05, k: int = 3) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed + 9000)
    Xp = X + rng.normal(0, sigma, size=X.shape)
    pred = (sigmoid(X @ w + b) >= 0.5)
    pred_p = (sigmoid(Xp @ w + b) >= 0.5)
    keep = pred == pred_p
    if explainer == 'exact_linear':
        Ap = exact_attributions(Xp, w, baseline)
    else:
        Ap = randomized_negative_control(exact_attributions(Xp, w, baseline), seed + 1)
    vals = np.array([jaccard(A[i], Ap[i], k) for i in range(len(X)) if keep[i]], dtype=float)
    return vals, float(keep.mean())


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    # Mann–Whitney definition; deterministic tie handling via average rank.
    order = np.argsort(scores, kind='mergesort')
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
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(values: np.ndarray, seed: int, reps: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manual_validation() -> dict:
    w = np.array([1.0, 0.5, 0.2])
    b = 0.0
    baseline = np.zeros(3)
    x = np.array([[2.0, 1.0, 1.0]])
    xp = np.array([[2.01, 0.98, 1.02]])
    a = exact_attributions(x, w, baseline)
    ap = exact_attributions(xp, w, baseline)
    fidelity = float(deletion_fidelity(x, a, w, b, baseline, kmax=2)[0])
    stability = jaccard(a[0], ap[0], k=2)
    assert np.allclose(a[0], [2.0, 0.5, 0.2])
    assert abs(fidelity - 2.25) < 1e-12
    assert abs(stability - 1.0) < 1e-12
    return {
        'fixture': {'x': x[0].tolist(), 'x_prime': xp[0].tolist(), 'weights': w.tolist(), 'baseline': baseline.tolist()},
        'attribution': a[0].tolist(),
        'fidelity_aopc_at_2_hand_expected': 2.25,
        'fidelity_aopc_at_2_computed': fidelity,
        'stability_jaccard_at_2_hand_expected': 1.0,
        'stability_jaccard_at_2_computed': stability,
        'status': 'passed',
    }


def run_pilot() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    model_rows = []
    start = time.perf_counter()
    for seed in SEEDS:
        X, y = generate(seed)
        Xtr, Xte, ytr, yte = split(X, y, seed)
        baseline = Xtr.mean(axis=0)
        t0 = time.perf_counter()
        w, b = fit_logistic(Xtr, ytr)
        fit_ms = (time.perf_counter() - t0) * 1000
        prob = sigmoid(Xte @ w + b)
        auc = auc_rank(yte, prob)
        acc = float(np.mean((prob >= .5) == yte))
        coef_cos = float(np.dot(w, BETA_TRUE) / (np.linalg.norm(w) * np.linalg.norm(BETA_TRUE)))
        model_rows.append({'seed': seed, 'accuracy': acc, 'roc_auc': auc, 'coefficient_cosine_to_truth': coef_cos, 'fit_ms': fit_ms})

        rng = np.random.default_rng(seed + 5000)
        order = rng.permutation(len(Xte))
        for n_eval in SAMPLE_SIZES:
            Xe = Xte[order[:n_eval]]
            exact = exact_attributions(Xe, w, baseline)
            methods = {
                'exact_linear': exact,
                'randomized_negative_control': randomized_negative_control(exact, seed + n_eval),
            }
            for method, A in methods.items():
                t1 = time.perf_counter()
                fidelity_vals = deletion_fidelity(Xe, A, w, b, baseline, kmax=3)
                stability_vals, preserved = stability_scores(Xe, A, w, b, baseline, method, seed + n_eval)
                runtime_ms = (time.perf_counter() - t1) * 1000
                f_lo, f_hi = bootstrap_ci(fidelity_vals, seed + n_eval)
                s_lo, s_hi = bootstrap_ci(stability_vals, seed + n_eval + 1)
                rows.extend([
                    {'seed': seed, 'sample_size': n_eval, 'explainer': method, 'metric': 'deletion_fidelity_aopc_at_3', 'estimate': float(np.mean(fidelity_vals)), 'ci_low': f_lo, 'ci_high': f_hi, 'n': len(fidelity_vals), 'prediction_preservation_rate': preserved, 'runtime_ms': runtime_ms},
                    {'seed': seed, 'sample_size': n_eval, 'explainer': method, 'metric': 'stability_top3_jaccard', 'estimate': float(np.mean(stability_vals)), 'ci_low': s_lo, 'ci_high': s_hi, 'n': len(stability_vals), 'prediction_preservation_rate': preserved, 'runtime_ms': runtime_ms},
                ])
    elapsed = time.perf_counter() - start
    return pd.DataFrame(rows), pd.DataFrame(model_rows), {'elapsed_seconds': elapsed}


def write_figures(results: pd.DataFrame) -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})
    colors = {'exact_linear': '#2783DE', 'randomized_negative_control': '#D5803B'}
    labels = {'exact_linear': 'Exact linear attribution', 'randomized_negative_control': 'Randomized negative control'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, metric, title, ylabel in [
        (axes[0], 'deletion_fidelity_aopc_at_3', 'Fidelity under top-feature deletion', 'Mean absolute logit change'),
        (axes[1], 'stability_top3_jaccard', 'Stability under prediction-preserving noise', 'Top-3 Jaccard similarity'),
    ]:
        sub = results[(results.metric == metric) & (results.sample_size == 200)]
        for method in colors:
            vals = sub[sub.explainer == method].estimate.to_numpy()
            mean = vals.mean(); lo, hi = np.quantile(vals, [.025, .975])
            x = 0 if method == 'exact_linear' else 1
            ax.errorbar(x, mean, yerr=[[mean-lo], [hi-mean]], fmt='o', color=colors[method], markersize=8, capsize=5, linewidth=2, label=labels[method])
        ax.set_xticks([0,1], ['Exact', 'Negative\ncontrol'])
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#E6E5E3', linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#F9F8F7')
    fig.suptitle('ExplainCheck synthetic pilot · 10 seeds · n=200 explanations per seed', x=.01, ha='left', fontsize=15, fontweight='bold', color='#2C2C2B')
    fig.savefig(FIGURES / 'fig-1-pilot-metrics.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'fig-1-pilot-metrics.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    var = (results.groupby(['sample_size','explainer','metric'])['estimate'].std().reset_index(name='sd_across_seeds'))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    markers = {'deletion_fidelity_aopc_at_3':'o', 'stability_top3_jaccard':'s'}
    for (method, metric), g in var.groupby(['explainer','metric']):
        ax.plot(g.sample_size, g.sd_across_seeds, marker=markers[metric], color=colors[method], linestyle='-' if metric.startswith('deletion') else '--', linewidth=2, markersize=6, label=f"{labels[method]} · {'fidelity' if metric.startswith('deletion') else 'stability'}")
    ax.set_title('Pilot variance decreases as evaluation samples increase', loc='left', fontsize=14, fontweight='bold')
    ax.set_xlabel('Explanations evaluated per seed')
    ax.set_ylabel('Standard deviation across seeds')
    ax.set_facecolor('#F9F8F7')
    ax.grid(color='#E6E5E3', linewidth=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
    fig.savefig(FIGURES / 'fig-2-variance-pilot.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'fig-2-variance-pilot.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_outputs(results: pd.DataFrame, models: pd.DataFrame, validation: dict, timing: dict) -> None:
    results.to_csv(ART / 'tidy-results.csv', index=False)
    models.to_csv(ART / 'model-performance.csv', index=False)
    summary = results.groupby(['sample_size','explainer','metric']).estimate.agg(['mean','std','min','max']).reset_index()
    summary.to_csv(TABLES / 'table-p1-pilot-summary.csv', index=False)
    variance = results.groupby(['sample_size','explainer','metric']).estimate.std().reset_index(name='sd_across_seeds')
    variance.to_csv(TABLES / 'table-p2-variance.csv', index=False)

    # Minimal LaTeX table generated from frozen CSV.
    final = summary[summary.sample_size == 200].copy()
    lines = ['\\begin{tabular}{llrr}', '\\toprule', 'Explainer & Metric & Mean & SD \\\\', '\\midrule']
    for _, r in final.iterrows():
        e = str(r.explainer).replace('_', '\\_')
        m = str(r.metric).replace('_', '\\_')
        lines.append(f"{e} & {m} & {r['mean']:.3f} & {r['std']:.3f} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    (TABLES / 'table-p1-pilot-summary.tex').write_text('\n'.join(lines), encoding='utf-8')

    model_summary = {c: {'mean': float(models[c].mean()), 'sd': float(models[c].std())} for c in ['accuracy','roc_auc','coefficient_cosine_to_truth','fit_ms']}
    result_records = []
    for _, r in final.iterrows():
        result_records.append({
            'explainer': r.explainer,
            'metric': r.metric,
            'sampleSize': int(r.sample_size),
            'estimate': float(r['mean']),
            'sdAcrossSeeds': float(r['std']),
        })
    benchmark = {
        'schemaVersion': '1.0.0',
        'protocolVersion': '1.0.0-rc1',
        'studyId': 'EC-SYNTH-LINEAR-PILOT-001',
        'status': 'pilot-not-confirmatory',
        'design': {
            'seeds': SEEDS,
            'sampleSizes': SAMPLE_SIZES,
            'features': FEATURES,
            'trueCoefficients': BETA_TRUE.tolist(),
            'nGeneratedPerSeed': 3000,
            'explainers': ['exact_linear', 'randomized_negative_control'],
            'metrics': ['deletion_fidelity_aopc_at_3', 'stability_top3_jaccard'],
        },
        'manualValidation': validation,
        'modelPerformance': model_summary,
        'resultsAtN200': result_records,
        'timing': timing,
        'limitations': [
            'Pilot uses a controlled linear-logit setting and does not establish performance on nonlinear or real-world data.',
            'Randomized attribution is a negative control, not a competitive explanation method.',
            'Pilot thresholds are engineering diagnostics and are not universal XAI standards.',
        ],
    }
    (ART / 'benchmark.json').write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
    (ART / 'manual-metric-validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')

    model_card = f"""# Model Card — Synthetic Linear Logistic Pilot

## Status
Pilot artifact; not suitable for deployment or substantive decision-making.

## Model details
- Model: custom L2-regularized logistic regression fitted by deterministic full-batch gradient descent
- Data: synthetic independent Gaussian features
- True coefficients: `{BETA_TRUE.tolist()}`
- Seeds: `{SEEDS}`

## Intended use
Validate ExplainCheck contracts, metric implementations, provenance, and report generation before adding external explainers or real-world datasets.

## Predictive performance across seeds
- Accuracy: {models.accuracy.mean():.3f} ± {models.accuracy.std():.3f}
- ROC AUC: {models.roc_auc.mean():.3f} ± {models.roc_auc.std():.3f}
- Coefficient cosine similarity to generating coefficients: {models.coefficient_cosine_to_truth.mean():.3f} ± {models.coefficient_cosine_to_truth.std():.3f}

## Explanation methods
- Exact linear attribution: learned coefficient × deviation from training mean
- Randomized negative control: within-sample permutation of exact attribution values

## Metrics
- Deletion fidelity AOPC@3: mean cumulative absolute logit change after masking top-ranked features to the training mean
- Stability Top-3 Jaccard: overlap of top features after small Gaussian perturbation, restricted to prediction-preserving pairs

## Limitations
This artifact validates machinery; it is not evidence that ExplainCheck outperforms Quantus or OpenXAI and is not a confirmatory result.
"""
    (ART / 'model-card.md').write_text(model_card, encoding='utf-8')

    methods = """# Generated Methods Snippet

For each of ten prespecified random seeds, 3,000 observations with eight independent standard-normal features were generated. Binary outcomes were sampled from a logistic model with coefficients (1.50, −1.20, 0.90, −0.70, 0, 0, 0, 0) and intercept −0.15. Data were divided into 80% training and 20% test partitions using a seed-specific deterministic permutation. An L2-regularized logistic-regression model was fitted by full-batch gradient descent. Local exact-linear attributions were defined as the learned coefficient multiplied by the feature’s deviation from the training-set mean. A within-instance randomized permutation of these attributions served as a negative control.

Deletion fidelity was operationalized as the mean cumulative absolute change in model logit after replacing the three highest-ranked features with their training-set means. Stability was operationalized as Top-3 Jaccard overlap after Gaussian input perturbation (σ=0.05), restricted to pairs whose predicted class was preserved. Metrics were estimated at evaluation sample sizes of 50, 100, and 200 per seed. The pilot reports across-seed variation and 1,000-replicate within-run bootstrap intervals; it is not a confirmatory hypothesis test.
"""
    (PAPER / 'methods-generated.md').write_text(methods, encoding='utf-8')

    n200 = final.pivot(index='explainer', columns='metric', values='mean')
    results_text = f"""# Pilot Results Snippet

Across ten seeds, the fitted logistic model achieved mean ROC AUC {models.roc_auc.mean():.3f} (SD {models.roc_auc.std():.3f}) and recovered the generating coefficient direction with cosine similarity {models.coefficient_cosine_to_truth.mean():.3f} (SD {models.coefficient_cosine_to_truth.std():.3f}). At 200 evaluated explanations per seed, exact linear attributions achieved mean deletion fidelity {n200.loc['exact_linear','deletion_fidelity_aopc_at_3']:.3f}, compared with {n200.loc['randomized_negative_control','deletion_fidelity_aopc_at_3']:.3f} for the randomized negative control. Mean Top-3 stability was {n200.loc['exact_linear','stability_top3_jaccard']:.3f} for exact attributions and {n200.loc['randomized_negative_control','stability_top3_jaccard']:.3f} for the negative control. These values validate the expected directional behavior of the pilot metrics but must not be interpreted as evidence about SHAP, LIME, or real-world deployments.
"""
    (PAPER / 'results-summary.md').write_text(results_text, encoding='utf-8')

    manifest = {
        'studyId': 'EC-SYNTH-LINEAR-PILOT-001',
        'createdWith': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': plt.matplotlib.__version__},
        'platform': platform.platform(),
        'command': 'python3 run_phase0.py',
        'files': {},
    }
    # Added after all current files exist.
    for path in sorted(ART.rglob('*')):
        if path.is_file() and path.name != 'run-manifest.json':
            manifest['files'][str(path.relative_to(ART))] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    (ART / 'run-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    validation = manual_validation()
    results, models, timing = run_pilot()
    write_figures(results)
    write_outputs(results, models, validation, timing)
    print(json.dumps({
        'status': 'ok',
        'manual_validation': validation['status'],
        'elapsed_seconds': timing['elapsed_seconds'],
        'roc_auc_mean': float(models.roc_auc.mean()),
        'artifact_dir': str(ART),
    }, indent=2))


if __name__ == '__main__':
    main()
