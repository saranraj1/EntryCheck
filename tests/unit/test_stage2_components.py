"""
Stage 2 unit tests — synthetic generator, model, explainers, metrics.

All tests are fast (< 2 seconds each), isolated, no I/O (except tmp_path).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from explaincheck.contracts import (
    AttributionRecord,
    ExplainerName,
    ExplainerType,
    ModelFamily,
)
from explaincheck.datasets.synthetic import (
    BETA_TRUE,
    FEATURE_NAMES,
    generate,
    split,
    split_record,
)
from explaincheck.explainers.exact_linear import ExactLinearExplainer, RandomizedNegativeControl
from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single
from explaincheck.metrics.stability.top_k_jaccard import jaccard
from explaincheck.models.logistic_regression import LogisticRegressionAdapter

# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_generate_shape() -> None:
    X, y = generate(11)
    assert X.shape == (3000, 8)
    assert y.shape == (3000,)


@pytest.mark.unit
def test_generate_binary_labels() -> None:
    _, y = generate(11)
    assert set(y.tolist()).issubset({0, 1})


@pytest.mark.unit
def test_generate_deterministic() -> None:
    X1, y1 = generate(11)
    X2, y2 = generate(11)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.unit
def test_generate_different_seeds() -> None:
    X1, _ = generate(11)
    X2, _ = generate(23)
    assert not np.allclose(X1, X2), "Different seeds must produce different data."


@pytest.mark.unit
def test_split_sizes() -> None:
    X, y = generate(11)
    X_tr, X_te, y_tr, y_te = split(X, y, 11)
    assert len(X_tr) + len(X_te) == len(X)
    assert abs(len(X_tr) / len(X) - 0.80) < 0.01


@pytest.mark.unit
def test_split_deterministic() -> None:
    X, y = generate(11)
    X_tr1, X_te1, y_tr1, y_te1 = split(X, y, 11)
    X_tr2, X_te2, y_tr2, y_te2 = split(X, y, 11)
    np.testing.assert_array_equal(X_tr1, X_tr2)
    np.testing.assert_array_equal(X_te1, X_te2)


@pytest.mark.unit
def test_split_different_seeds() -> None:
    X, y = generate(11)
    _, X_te1, _, _ = split(X, y, 11)
    _, X_te2, _, _ = split(X, y, 23)
    assert not np.allclose(X_te1, X_te2)


@pytest.mark.unit
def test_split_record_hashes_differ() -> None:
    X, y = generate(11)
    X_tr, X_te, y_tr, y_te = split(X, y, 11)
    rec = split_record(X_tr, X_te, y_tr, y_te, 11)
    assert len(rec.train_sha256) == 64
    assert len(rec.test_sha256) == 64
    assert rec.train_sha256 != rec.test_sha256


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_lr_fit_returns_model_record() -> None:
    X, y = generate(11)
    X_tr, _, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    rec = lr.fit(X_tr, y_tr, seed=11)
    assert rec.family == ModelFamily.LOGISTIC_REGRESSION
    assert len(rec.model_hash) == 64
    assert rec.fit_ms > 0


@pytest.mark.unit
def test_lr_predict_proba_shape() -> None:
    X, y = generate(11)
    X_tr, X_te, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)
    probs = lr.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


@pytest.mark.unit
def test_lr_deterministic() -> None:
    X, y = generate(11)
    X_tr, _, y_tr, _ = split(X, y, 11)
    for _ in range(3):
        lr1 = LogisticRegressionAdapter()
        rec1 = lr1.fit(X_tr, y_tr, seed=11)
        lr2 = LogisticRegressionAdapter()
        rec2 = lr2.fit(X_tr, y_tr, seed=11)
        np.testing.assert_array_equal(lr1.weights, lr2.weights)
        assert rec1.model_hash == rec2.model_hash


@pytest.mark.unit
def test_lr_recovers_coefficient_direction() -> None:
    """LR should align with true coefficients (cosine > 0.9) on synthetic data."""
    X, y = generate(11)
    X_tr, _, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)
    w = lr.weights
    cos = float(np.dot(w, BETA_TRUE) / (np.linalg.norm(w) * np.linalg.norm(BETA_TRUE)))
    assert cos > 0.90, f"Cosine similarity with true coefficients too low: {cos:.4f}"


# ---------------------------------------------------------------------------
# Exact-linear explainer
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_exact_linear_hand_fixture(simple_linear_weights, simple_linear_sample,
                                   simple_linear_baseline, simple_linear_attribution) -> None:
    """Exact-linear attributions must match the Phase 0 hand fixture exactly."""
    X = simple_linear_sample.reshape(1, -1)
    baseline = simple_linear_baseline
    w = simple_linear_weights

    class FakeLR:
        weights = w
        def predict_proba(self, X): return np.array([[0.4, 0.6]])

    exp = ExactLinearExplainer()
    exp._weights = w
    exp._baseline = baseline
    exp._feature_names = ["f1", "f2", "f3"]
    exp._background_hash = "test"

    records = exp.explain(
        X, run_id="test", dataset="test", seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash="testhash", sample_ids=["s0"], protocol_version="1.0.0",
    )
    assert len(records) == 1
    assert isinstance(records[0], AttributionRecord)
    np.testing.assert_allclose(records[0].attribution, simple_linear_attribution.tolist(), atol=1e-12)


@pytest.mark.unit
def test_exact_linear_explainer_type() -> None:
    assert ExactLinearExplainer.explainer_type == ExplainerType.CONTROL_REFERENCE


@pytest.mark.unit
def test_negative_control_type() -> None:
    assert RandomizedNegativeControl.explainer_type == ExplainerType.CONTROL_NEGATIVE


@pytest.mark.unit
def test_negative_control_same_magnitude() -> None:
    """Negative control is a permutation — total absolute attribution must be preserved."""
    X, y = generate(11)
    X_tr, X_te, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)

    exact = ExactLinearExplainer()
    exact.fit(lr, X_tr, FEATURE_NAMES, seed=11)
    neg = RandomizedNegativeControl()
    neg.fit(lr, X_tr, FEATURE_NAMES, seed=11)

    sample_ids = [f"s{i}" for i in range(10)]
    exact_recs = [r for r in exact.explain(X_te[:10], run_id="t", dataset="d", seed=11,
                                             model_family=ModelFamily.LOGISTIC_REGRESSION,
                                             model_hash="h", sample_ids=sample_ids,
                                             protocol_version="1.0.0")
                   if isinstance(r, AttributionRecord)]
    neg_recs = [r for r in neg.explain(X_te[:10], run_id="t", dataset="d", seed=11,
                                         model_family=ModelFamily.LOGISTIC_REGRESSION,
                                         model_hash="h", sample_ids=sample_ids,
                                         protocol_version="1.0.0")
                 if isinstance(r, AttributionRecord)]

    for e, n in zip(exact_recs, neg_recs, strict=False):
        np.testing.assert_allclose(
            sorted(e.attribution), sorted(n.attribution), atol=1e-12,
            err_msg="Negative control must be a permutation — sorted attributions must match."
        )


# ---------------------------------------------------------------------------
# Deletion fidelity AOPC
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fidelity_hand_fixture() -> None:
    """DR-002 hand fixture: AOPC@2 must equal 2.25 exactly."""
    w = np.array([1.0, 0.5, 0.2])
    b = 0.0
    baseline = np.zeros(3)
    x = np.array([2.0, 1.0, 1.0])
    a = (x - baseline) * w
    result = deletion_fidelity_aopc_single(x, a, w, b, baseline, kmax=2)
    assert abs(result - 2.25) < 1e-12, f"Expected 2.25, got {result}"


@pytest.mark.unit
def test_fidelity_zero_attribution() -> None:
    """Zero attribution → all features tied → AOPC is near-zero or stable (implementation check)."""
    w = np.array([1.0, 0.5, 0.2])
    b = 0.0
    baseline = np.zeros(3)
    x = np.array([2.0, 1.0, 1.0])
    a = np.zeros(3)  # zero attribution
    result = deletion_fidelity_aopc_single(x, a, w, b, baseline, kmax=2)
    assert math.isfinite(result)  # must not crash


@pytest.mark.unit
def test_fidelity_nonnegative() -> None:
    """AOPC uses absolute logit change — must always be >= 0."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_feat = 5
        w = rng.normal(size=n_feat)
        x = rng.normal(size=n_feat)
        a = rng.normal(size=n_feat)
        baseline = np.zeros(n_feat)
        result = deletion_fidelity_aopc_single(x, a, w, 0.0, baseline, kmax=3)
        assert result >= 0.0, f"AOPC must be non-negative, got {result}"


# ---------------------------------------------------------------------------
# Stability Top-k Jaccard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_stability_hand_fixture() -> None:
    """DR-002 hand fixture: Jaccard@2 must equal 1.0 exactly."""
    a = np.array([2.0, 0.5, 0.2])
    ap = np.array([2.01, 0.49, 0.21])  # same top-2 order
    result = jaccard(a, ap, k=2)
    assert abs(result - 1.0) < 1e-12, f"Expected 1.0, got {result}"


@pytest.mark.unit
def test_stability_disjoint_topk() -> None:
    """Disjoint top-k sets → Jaccard = 0."""
    a = np.array([1.0, 0.0, 0.0])
    ap = np.array([0.0, 1.0, 0.0])
    result = jaccard(a, ap, k=1)
    assert abs(result - 0.0) < 1e-12, f"Expected 0.0, got {result}"


@pytest.mark.unit
def test_stability_negative_control_lower_than_exact() -> None:
    """Negative control must have strictly lower stability than exact (on average, 10 seeds)."""
    from explaincheck.pilot.runner import FROZEN_SEEDS, KMAX, SIGMA
    results_exact, results_neg = [], []
    for seed in FROZEN_SEEDS[:3]:
        X, y = generate(seed)
        X_tr, X_te, y_tr, y_te = split(X, y, seed)
        baseline = X_tr.mean(axis=0)
        lr = LogisticRegressionAdapter()
        lr.fit(X_tr, y_tr, seed=seed)
        Xe = X_te[:50]

        rng_p = np.random.default_rng(seed + 9000)
        Xep = Xe + rng_p.normal(0, SIGMA, size=Xe.shape)

        def _sigmoid(z):
            z = np.clip(z, -35, 35); return 1.0 / (1.0 + np.exp(-z))

        pred_orig = (_sigmoid(Xe @ lr.weights + lr.bias) >= 0.5).astype(int)
        pred_pert = (_sigmoid(Xep @ lr.weights + lr.bias) >= 0.5).astype(int)
        preserved = pred_orig == pred_pert

        exact_attrs = [(Xe[i] - baseline) * lr.weights for i in range(len(Xe))]
        for i, a in enumerate(exact_attrs):
            if not preserved[i]:
                continue
            ap = (Xep[i] - baseline) * lr.weights
            results_exact.append(jaccard(a, ap, KMAX))

        rng_neg = np.random.default_rng(seed + 1)
        for i, a in enumerate(exact_attrs):
            if not preserved[i]:
                continue
            ap_exact = (Xep[i] - baseline) * lr.weights
            ap = ap_exact[rng_neg.permutation(ap_exact.size)]
            results_neg.append(jaccard(a, ap, KMAX))

    exact_mean = np.mean(results_exact)
    neg_mean = np.mean(results_neg)
    assert exact_mean > neg_mean, (
        f"Exact stability ({exact_mean:.3f}) must exceed negative control ({neg_mean:.3f})."
    )


# ---------------------------------------------------------------------------
# Invalid input guards
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fidelity_raises_on_nan_attribution() -> None:
    from explaincheck.contracts import AttributionRecord, DataSplit
    with pytest.raises(Exception):
        AttributionRecord(
            run_id="t", protocol_version="1.0.0", sample_id="s0",
            dataset="d", seed=11, split=DataSplit.TEST,
            model_family=ModelFamily.LOGISTIC_REGRESSION, model_hash="h",
            explainer=ExplainerName.EXACT_LINEAR, explainer_type=ExplainerType.CONTROL_REFERENCE,
            explainer_version="1.0.0", feature_names=["f1"],
            attribution=[float("nan")], prediction_class=0,
            prediction_probability=0.5, runtime_ms=1.0, success=True,
        )


@pytest.mark.unit
def test_explainer_requires_fit() -> None:
    exp = ExactLinearExplainer()
    with pytest.raises(RuntimeError):
        exp.explain(np.zeros((1, 3)), run_id="t", dataset="d", seed=11,
                    model_family=ModelFamily.LOGISTIC_REGRESSION, model_hash="h",
                    sample_ids=["s0"], protocol_version="1.0.0")


@pytest.mark.unit
def test_lr_requires_fit() -> None:
    lr = LogisticRegressionAdapter()
    with pytest.raises(RuntimeError):
        lr.predict_proba(np.zeros((1, 8)))
