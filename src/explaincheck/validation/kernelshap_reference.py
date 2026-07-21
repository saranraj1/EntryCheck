"""
ExplainCheck — Stage 3 KernelSHAP multi-seed reference implementation.

Scientific fixture (frozen per DR-006A §3):
    f(x) = 0.3 + X @ [2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0]
    Background: 50 training rows (seeded), nsamples=512
    Seeds: 0, 1, 2, 3, 4

Frozen gates (applied to 5-seed mean):
    Mean cosine similarity >= 0.99
    Mean Spearman correlation >= 0.95
    Mean nonzero-feature sign agreement >= 0.95

This module provides:
    - run_kernelshap_seed(seed): run KernelSHAP for one seed, return per-seed metrics
    - FROZEN_WEIGHTS: the exact weight vector used in the fixture
    - FIXTURE_BIAS: scalar bias
"""

from __future__ import annotations

import time

import numpy as np
import shap
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Fixture definition (frozen per DR-006A §3)
# ---------------------------------------------------------------------------

FROZEN_WEIGHTS: np.ndarray = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
FIXTURE_BIAS: float = 0.3
N_FEATURES: int = 8
N_BACKGROUND: int = 50
N_TEST: int = 30
N_SAMPLES_SHAP: int = 512


def _generate_fixture_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate background and test matrices for KernelSHAP validation.

    Background rows are drawn from N(0,1) using `seed` for reproducibility.
    Test rows are drawn from N(0,1) using `seed + 1000`.
    """
    rng_bg = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1000)
    X_background = rng_bg.standard_normal((N_BACKGROUND, N_FEATURES))
    X_test = rng_test.standard_normal((N_TEST, N_FEATURES))
    return X_background, X_test


def _exact_attribution(X: np.ndarray) -> np.ndarray:
    """
    Compute exact linear attribution for the fixture model.
    A_i = w_j * (x_ij - mean_background_j) for each feature j.
    Returns array of shape (n_samples, n_features).

    Note: For KernelSHAP validation we compare against a wrapped sklearn
    model so that KernelSHAP treats it as a black box.
    """
    raise NotImplementedError("Use run_kernelshap_seed() which uses sklearn wrapper.")


def _predict_fn(X_background: np.ndarray) -> object:
    """Return a predict function wrapping the fixture linear model."""

    def predict(X: np.ndarray) -> np.ndarray:
        return X @ FROZEN_WEIGHTS + FIXTURE_BIAS

    return predict


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fraction of nonzero-in-truth features where SHAP and truth agree on sign.

    'Nonzero in truth' = FROZEN_WEIGHTS != 0 (features 0–4).
    """
    nonzero_mask = FROZEN_WEIGHTS != 0.0
    if not np.any(nonzero_mask):
        return 1.0
    a_nz = a[nonzero_mask]
    b_nz = b[nonzero_mask]
    # Agreement: same nonzero sign, or both near zero
    agree = np.sign(a_nz) == np.sign(b_nz)
    return float(np.mean(agree))


def run_kernelshap_seed(seed: int) -> dict:
    """
    Run KernelSHAP validation for one seed. Returns a metrics dict.

    Metrics returned (all over N_TEST samples):
        seed:              int
        mean_cosine:       float  ← gate: >= 0.99
        mean_spearman:     float  ← gate: >= 0.95
        mean_sign_agree:   float  ← gate: >= 0.95
        mean_mae:          float  (descriptive)
        max_mae:           float  (descriptive)
        mean_topk_agree:   float  (descriptive, k=3)
        runtime_ms:        float  (descriptive)
        gate_cosine:       bool
        gate_spearman:     bool
        gate_sign_agree:   bool
        all_gates_pass:    bool
    """
    X_background, X_test = _generate_fixture_data(seed)
    predict = _predict_fn(X_background)

    # KernelSHAP explanation
    t0 = time.perf_counter()
    explainer = shap.KernelExplainer(predict, X_background, silent=True)
    shap_values = explainer.shap_values(X_test, nsamples=N_SAMPLES_SHAP, l1_reg=0)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Exact attribution: A_j = w_j * (x_j - E[x_j])
    X_bg_mean = X_background.mean(axis=0)
    exact = (X_test - X_bg_mean) * FROZEN_WEIGHTS  # (N_TEST, N_FEATURES)

    # Per-sample metrics
    cosines, spearmans, signs, maes, topks = [], [], [], [], []
    for i in range(N_TEST):
        s = shap_values[i]
        e = exact[i]
        cosines.append(cosine_sim(s, e))
        rho, _ = spearmanr(s, e)
        spearmans.append(float(rho) if np.isfinite(rho) else float("nan"))
        signs.append(sign_agreement(s, e))
        maes.append(float(np.mean(np.abs(s - e))))
        # Top-k agreement (k=3)
        top_s = set(np.argsort(-np.abs(s))[:3].tolist())
        top_e = set(np.argsort(-np.abs(e))[:3].tolist())
        topks.append(len(top_s & top_e) / len(top_s | top_e) if (top_s | top_e) else 1.0)

    mean_cosine = float(np.nanmean(cosines))
    mean_spearman = float(np.nanmean(spearmans))
    mean_sign_agree = float(np.nanmean(signs))
    mean_mae = float(np.nanmean(maes))
    max_mae = float(np.nanmax(np.abs(np.array([shap_values[i] - exact[i] for i in range(N_TEST)]))))
    mean_topk = float(np.nanmean(topks))

    return {
        "seed": seed,
        "mean_cosine": mean_cosine,
        "mean_spearman": mean_spearman,
        "mean_sign_agree": mean_sign_agree,
        "mean_mae": mean_mae,
        "max_mae": max_mae,
        "mean_topk_agree": mean_topk,
        "runtime_ms": runtime_ms,
        "gate_cosine": mean_cosine >= 0.99,
        "gate_spearman": mean_spearman >= 0.95,
        "gate_sign_agree": mean_sign_agree >= 0.95,
        "all_gates_pass": (
            mean_cosine >= 0.99 and mean_spearman >= 0.95 and mean_sign_agree >= 0.95
        ),
    }
