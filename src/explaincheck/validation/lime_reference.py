"""
ExplainCheck — Stage 3 LIME multi-seed reference implementation.

Scientific fixture (frozen per DR-006A §3):
    f(x) = 0.3 + X @ [2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0]
    Mode: regression, discretize_continuous=False
    Background: 200 training rows (seeded), num_samples=512
    Seeds: 0, 1, 2, 3, 4

Frozen gates (applied to 5-seed mean):
    Mean cosine similarity >= 0.95
    Mean nonzero-feature sign agreement >= 0.90
    Mean Top-k signal recall >= 0.90  (k=3, nonzero ground-truth features)

Descriptive (no frozen gate):
    Spearman rank correlation
    Dense attribution variation across seeds
    Actual numeric kernel width (not just "auto")
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Fixture definition (frozen per DR-006A §3)
# ---------------------------------------------------------------------------

FROZEN_WEIGHTS: np.ndarray = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
FIXTURE_BIAS: float = 0.3
N_FEATURES: int = 8
N_BACKGROUND: int = 200  # LIME uses larger background
N_TEST: int = 30
NUM_SAMPLES_LIME: int = 512
FEATURE_NAMES: list[str] = [f"f{i}" for i in range(N_FEATURES)]

# Ground-truth nonzero feature indices (features 0–4)
NONZERO_IDX: list[int] = [0, 1, 2, 3, 4]


def _generate_fixture_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate background and test matrices for LIME validation."""
    rng_bg = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1000)
    X_background = rng_bg.standard_normal((N_BACKGROUND, N_FEATURES))
    X_test = rng_test.standard_normal((N_TEST, N_FEATURES))
    return X_background, X_test


def _predict_fn(X: np.ndarray) -> np.ndarray:
    """Predict using the fixture linear model."""
    return X @ FROZEN_WEIGHTS + FIXTURE_BIAS


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of nonzero-in-truth features where LIME and truth agree on sign."""
    a_nz = a[NONZERO_IDX]
    b_nz = b[NONZERO_IDX]
    agree = np.sign(a_nz) == np.sign(b_nz)
    return float(np.mean(agree))


def topk_signal_recall(a: np.ndarray, k: int = 3) -> float:
    """
    Top-k signal recall: fraction of ground-truth top-k features
    that appear in LIME's top-k features.

    Ground-truth top-k = features with largest |FROZEN_WEIGHTS|.
    """
    gt_topk = set(np.argsort(-np.abs(FROZEN_WEIGHTS))[:k].tolist())
    pred_topk = set(np.argsort(-np.abs(a))[:k].tolist())
    if not gt_topk:
        return 1.0
    return len(gt_topk & pred_topk) / len(gt_topk)


def run_lime_seed(seed: int) -> dict:
    """
    Run LIME validation for one seed. Returns a metrics dict.

    Metrics returned (all averaged over N_TEST samples):
        seed:              int
        mean_cosine:       float  ← gate: >= 0.95
        mean_sign_agree:   float  ← gate: >= 0.90
        mean_topk_recall:  float  ← gate: >= 0.90 (k=3)
        mean_spearman:     float  (descriptive)
        std_attribution:   float  (descriptive — variation across seeds via std of mean attributions)
        kernel_width:      float  (actual numeric width used)
        runtime_ms:        float  (descriptive)
        gate_cosine:       bool
        gate_sign_agree:   bool
        gate_topk_recall:  bool
        all_gates_pass:    bool
    """
    X_background, X_test = _generate_fixture_data(seed)

    # Ground truth for LIME: the true model weights (LIME approximates local gradient)
    # For a linear model f(x) = w·x + b, LIME's local linear coefficients ≈ w_j
    # This is different from Shapley attributions (which are w_j * (x_j - E[x_j]))
    ground_truth = FROZEN_WEIGHTS  # shape (N_FEATURES,)

    # LIME explainer — regression mode, no discretization
    explainer = LimeTabularExplainer(
        X_background,
        feature_names=FEATURE_NAMES,
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    # Resolve actual kernel width (LIME default: sqrt(n_features * 0.75))
    kernel_width = float(np.sqrt(N_FEATURES * 0.75))

    t0 = time.perf_counter()
    lime_attrs = np.zeros((N_TEST, N_FEATURES))
    for i in range(N_TEST):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanation = explainer.explain_instance(
                X_test[i],
                _predict_fn,
                num_features=N_FEATURES,
                num_samples=NUM_SAMPLES_LIME,
            )
        # Map LIME coefs back to feature order
        coef_map = dict(explanation.as_list())
        for j, name in enumerate(FEATURE_NAMES):
            lime_attrs[i, j] = coef_map.get(name, 0.0)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Per-sample metrics — compare each sample's LIME coefficients to ground truth weights
    cosines, signs, topks, spearmans = [], [], [], []
    for i in range(N_TEST):
        s = lime_attrs[i]
        e = ground_truth  # compare LIME local coefs against true weights
        cosines.append(cosine_sim(s, e))
        signs.append(sign_agreement(s, e))
        topks.append(topk_signal_recall(s, k=3))
        rho, _ = spearmanr(s, e)
        spearmans.append(float(rho) if np.isfinite(rho) else float("nan"))

    mean_cosine = float(np.nanmean(cosines))
    mean_sign_agree = float(np.nanmean(signs))
    mean_topk_recall = float(np.nanmean(topks))
    mean_spearman = float(np.nanmean(spearmans))
    std_attribution = float(np.std(lime_attrs.mean(axis=0)))

    return {
        "seed": seed,
        "mean_cosine": mean_cosine,
        "mean_sign_agree": mean_sign_agree,
        "mean_topk_recall": mean_topk_recall,
        "mean_spearman": mean_spearman,
        "std_attribution": std_attribution,
        "kernel_width": kernel_width,
        "runtime_ms": runtime_ms,
        "gate_cosine": mean_cosine >= 0.95,
        "gate_sign_agree": mean_sign_agree >= 0.90,
        "gate_topk_recall": mean_topk_recall >= 0.90,
        "all_gates_pass": (
            mean_cosine >= 0.95 and mean_sign_agree >= 0.90 and mean_topk_recall >= 0.90
        ),
    }
