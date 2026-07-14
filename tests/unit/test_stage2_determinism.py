"""
Stage 2 determinism tests — same inputs → same outputs on multiple calls.

Flaky tests (non-deterministic results) would invalidate reproducibility claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from explaincheck.contracts import ModelFamily
from explaincheck.datasets.synthetic import FEATURE_NAMES, generate, split
from explaincheck.explainers.exact_linear import ExactLinearExplainer, RandomizedNegativeControl
from explaincheck.models.logistic_regression import LogisticRegressionAdapter


@pytest.mark.unit
def test_synthetic_generator_determinism() -> None:
    """generate() must produce identical output across repeated calls with same seed."""
    for seed in [11, 23, 37]:
        X1, y1 = generate(seed)
        X2, y2 = generate(seed)
        np.testing.assert_array_equal(X1, X2, err_msg=f"X mismatch at seed={seed}")
        np.testing.assert_array_equal(y1, y2, err_msg=f"y mismatch at seed={seed}")


@pytest.mark.unit
def test_split_determinism() -> None:
    """split() must be reproducible for the same data and seed."""
    X, y = generate(11)
    for seed in [11, 23]:
        X_tr1, X_te1, y_tr1, y_te1 = split(X, y, seed)
        X_tr2, X_te2, y_tr2, y_te2 = split(X, y, seed)
        np.testing.assert_array_equal(X_tr1, X_tr2)
        np.testing.assert_array_equal(X_te1, X_te2)


@pytest.mark.unit
def test_lr_determinism() -> None:
    """LR fitted on same data and seed must produce identical weights."""
    X, y = generate(11)
    X_tr, _, y_tr, _ = split(X, y, 11)
    for _ in range(3):
        lr = LogisticRegressionAdapter()
        lr.fit(X_tr, y_tr, seed=11)
        w1 = lr.weights.copy()
        lr2 = LogisticRegressionAdapter()
        lr2.fit(X_tr, y_tr, seed=11)
        np.testing.assert_array_equal(w1, lr2.weights)


@pytest.mark.unit
def test_exact_linear_determinism() -> None:
    """Exact-linear attributions must be identical across runs."""
    X, y = generate(11)
    X_tr, X_te, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)

    def _explain():
        exp = ExactLinearExplainer()
        exp.fit(lr, X_tr, FEATURE_NAMES, seed=11)
        records = exp.explain(
            X_te[:5], run_id="t", dataset="d", seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION, model_hash="h",
            sample_ids=[f"s{i}" for i in range(5)], protocol_version="1.0.0"
        )
        return [r.attribution for r in records]

    a1 = _explain()
    a2 = _explain()
    for i, (v1, v2) in enumerate(zip(a1, a2, strict=False)):
        np.testing.assert_array_equal(v1, v2, err_msg=f"Mismatch at sample {i}")


@pytest.mark.unit
def test_negative_control_seed_determinism() -> None:
    """Negative control must produce identical outputs for same seed."""
    X, y = generate(11)
    X_tr, X_te, y_tr, _ = split(X, y, 11)
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)

    def _explain():
        neg = RandomizedNegativeControl()
        neg.fit(lr, X_tr, FEATURE_NAMES, seed=11)
        records = neg.explain(
            X_te[:5], run_id="t", dataset="d", seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION, model_hash="h",
            sample_ids=[f"s{i}" for i in range(5)], protocol_version="1.0.0"
        )
        return [r.attribution for r in records]

    a1 = _explain()
    a2 = _explain()
    for i, (v1, v2) in enumerate(zip(a1, a2, strict=False)):
        np.testing.assert_array_equal(v1, v2, err_msg=f"Mismatch at sample {i}")


@pytest.mark.unit
def test_fidelity_metric_determinism() -> None:
    """Fidelity AOPC must be identical across repeated calls."""
    from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single
    rng = np.random.default_rng(42)
    w = rng.normal(size=5)
    x = rng.normal(size=5)
    baseline = rng.normal(size=5)
    a = rng.normal(size=5)
    v1 = deletion_fidelity_aopc_single(x, a, w, 0.0, baseline, kmax=3)
    v2 = deletion_fidelity_aopc_single(x, a, w, 0.0, baseline, kmax=3)
    assert v1 == v2


@pytest.mark.unit
def test_stability_metric_determinism() -> None:
    """Jaccard stability must be identical across repeated calls."""
    from explaincheck.metrics.stability.top_k_jaccard import jaccard
    rng = np.random.default_rng(42)
    a = rng.normal(size=8)
    ap = rng.normal(size=8)
    v1 = jaccard(a, ap, k=3)
    v2 = jaccard(a, ap, k=3)
    assert v1 == v2
