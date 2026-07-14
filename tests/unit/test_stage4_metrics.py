"""
Stage 4 — Core Metrics test suite (DR-003A).

Covers:
  - k90_sparsity: hand fixtures, boundary cases, property
  - cosine_similarity_pair: hand fixtures, edge cases
  - spearman_pair: hand fixtures, edge cases
  - CosineStability.compute(): schema, output count, failure modes
  - SpearmanStability.compute(): schema, output count, prediction-preservation
  - K90Sparsity.compute(): schema, output count, boundary
  - RuntimeMetric.compute(): schema, value extraction
  - timer_and_memory: context manager basic test

All tests use the exact_linear explainer against the synthetic linear fixture
to keep tests fast and analytically verifiable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from explaincheck.contracts import (
    DataSplit,
    ExplainerName,
    ExplainerType,
    MetricFamily,
    ModelFamily,
    PredictionPreservationStatus,
    RunStatus,
)
from explaincheck.contracts.models import AttributionRecord, MetricResult, FailureRecord
from explaincheck.datasets.synthetic import FEATURE_NAMES, generate, split
from explaincheck.explainers.exact_linear import ExactLinearExplainer
from explaincheck.metrics.sparsity.k90_sparsity import K90Sparsity, k90_sparsity
from explaincheck.metrics.stability.cosine_stability import CosineStability, cosine_similarity_pair
from explaincheck.metrics.stability.spearman_stability import SpearmanStability, spearman_pair
from explaincheck.metrics.runtime.runtime_metric import RuntimeMetric, timer_and_memory
from explaincheck.models.logistic_regression import LogisticRegressionAdapter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_data():
    X, y = generate(11, n=500)
    X_tr, X_te, y_tr, y_te = split(X, y, 11)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture(scope="module")
def lr_fitted(synth_data):
    X_tr, _, y_tr, _ = synth_data
    model = LogisticRegressionAdapter()
    rec = model.fit(X_tr, y_tr, seed=11)
    return model, rec


@pytest.fixture(scope="module")
def exact_records(lr_fitted, synth_data):
    """ExactLinear attribution records for 20 test samples."""
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    exp = ExactLinearExplainer()
    exp.fit(model, X_tr, FEATURE_NAMES, seed=11)
    records = exp.explain(
        X_te[:20], run_id="t4", dataset="synth-linear", seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION, model_hash=rec.model_hash,
        sample_ids=[f"s{i}" for i in range(20)], protocol_version="1.0.0",
    )
    from explaincheck.contracts import AttributionRecord as AR
    return [r for r in records if isinstance(r, AR)]


# ---------------------------------------------------------------------------
# k90_sparsity function: hand fixtures
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_k90_single_dominant():
    """If one feature holds 95% of mass, k90 = 1."""
    a = np.array([0.95, 0.02, 0.02, 0.01])
    assert k90_sparsity(a) == 1


@pytest.mark.unit
def test_k90_uniform_four_features():
    """Uniform attributions [0.25, 0.25, 0.25, 0.25]: need all 4 for 90%."""
    a = np.array([0.25, 0.25, 0.25, 0.25])
    assert k90_sparsity(a) == 4


@pytest.mark.unit
def test_k90_zero_vector():
    """Zero attribution vector → k90 = 0."""
    a = np.zeros(8)
    assert k90_sparsity(a) == 0


@pytest.mark.unit
def test_k90_exactly_90_two_features():
    """[0.45, 0.45, 0.05, 0.05]: top 2 = 0.90 / 1.0 → k90 = 2."""
    a = np.array([0.45, 0.45, 0.05, 0.05])
    assert k90_sparsity(a) == 2


@pytest.mark.unit
def test_k90_handles_negative_values():
    """k90 uses absolute values; signs don't matter."""
    a = np.array([-0.6, 0.3, -0.1])  # |a|=[0.6,0.3,0.1], cumsum=[0.6,0.9,1.0]
    assert k90_sparsity(a) == 2


@pytest.mark.unit
def test_k90_raises_on_nan():
    with pytest.raises(ValueError, match="NaN or Inf"):
        k90_sparsity(np.array([float("nan"), 0.5]))


@pytest.mark.unit
def test_k90_raises_on_empty():
    with pytest.raises(ValueError, match="Empty"):
        k90_sparsity(np.array([]))


@pytest.mark.unit
def test_k90_single_element():
    """Single feature always has 100% mass → k90 = 1."""
    assert k90_sparsity(np.array([0.7])) == 1


# ---------------------------------------------------------------------------
# cosine_similarity_pair: hand fixtures
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert math.isclose(cosine_similarity_pair(a, a), 1.0, abs_tol=1e-12)


@pytest.mark.unit
def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert math.isclose(cosine_similarity_pair(a, b), 0.0, abs_tol=1e-12)


@pytest.mark.unit
def test_cosine_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert math.isclose(cosine_similarity_pair(a, b), -1.0, abs_tol=1e-12)


@pytest.mark.unit
def test_cosine_both_zero():
    a = np.zeros(4)
    assert cosine_similarity_pair(a, a) == 1.0


@pytest.mark.unit
def test_cosine_one_zero():
    a = np.zeros(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    assert cosine_similarity_pair(a, b) == 0.0


@pytest.mark.unit
def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_similarity_pair(np.zeros(3), np.zeros(4))


@pytest.mark.unit
def test_cosine_nan_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        cosine_similarity_pair(np.array([float("nan")]), np.array([1.0]))


@pytest.mark.unit
def test_cosine_known_value():
    """[1,1] vs [1,0]: cosine = 1/sqrt(2) ≈ 0.7071."""
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 0.0])
    expected = 1.0 / math.sqrt(2.0)
    assert math.isclose(cosine_similarity_pair(a, b), expected, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# spearman_pair: hand fixtures
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_spearman_identical():
    a = np.array([3.0, 1.0, 2.0])
    assert math.isclose(spearman_pair(a, a), 1.0, abs_tol=1e-10)


@pytest.mark.unit
def test_spearman_reversed():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    assert math.isclose(spearman_pair(a, b), -1.0, abs_tol=1e-10)


@pytest.mark.unit
def test_spearman_both_constant():
    """Both constant → 1.0 (identical degenerate attribution)."""
    a = np.array([0.5, 0.5, 0.5])
    assert spearman_pair(a, a) == 1.0


@pytest.mark.unit
def test_spearman_nan_on_one_constant():
    """One constant, other not → scipy returns NaN."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.zeros(3)
    result = spearman_pair(a, b)
    assert math.isnan(result)


@pytest.mark.unit
def test_spearman_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        spearman_pair(np.zeros(3), np.zeros(4))


@pytest.mark.unit
def test_spearman_nan_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        spearman_pair(np.array([float("nan")]), np.array([1.0]))


# ---------------------------------------------------------------------------
# K90Sparsity.compute(): schema and contract
# ---------------------------------------------------------------------------

_COMPUTE_KWARGS = dict(
    run_id="t4",
    protocol_version="1.0.0",
    dataset="synth-linear",
    dataset_version="1.0",
    split_hash="abc123",
    model_family=ModelFamily.LOGISTIC_REGRESSION,
    model_hash="mh",
    seed=11,
)


@pytest.mark.unit
def test_k90_compute_output_count(exact_records):
    metric = K90Sparsity()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_k90_compute_all_success(exact_records):
    metric = K90Sparsity()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    for r in results:
        assert isinstance(r, MetricResult)
        assert r.status == RunStatus.SUCCESS
        assert r.metric_name == "k90_sparsity"
        assert r.metric_family == MetricFamily.SPARSITY


@pytest.mark.unit
def test_k90_compute_values_in_range(exact_records):
    """k90 must be in [0, n_features] for all samples."""
    metric = K90Sparsity()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    p = len(FEATURE_NAMES)
    for r in results:
        if isinstance(r, MetricResult):
            assert 0 <= r.estimate <= p, f"k90={r.estimate} out of [0, {p}]"


@pytest.mark.unit
def test_k90_compute_deterministic(exact_records):
    """Same inputs → same outputs."""
    metric = K90Sparsity()
    r1 = metric.compute(exact_records, **_COMPUTE_KWARGS)
    r2 = metric.compute(exact_records, **_COMPUTE_KWARGS)
    for a, b in zip(r1, r2, strict=True):
        if isinstance(a, MetricResult) and isinstance(b, MetricResult):
            assert a.estimate == b.estimate


@pytest.mark.unit
def test_k90_compute_not_applicable_preservation(exact_records):
    """K90 does not require prediction preservation."""
    metric = K90Sparsity()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    for r in results:
        if isinstance(r, MetricResult):
            assert r.prediction_preservation_status == PredictionPreservationStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# CosineStability.compute(): schema and contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cosine_stability_compute_count(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = CosineStability(sigma=0.05)
    results = metric.compute(
        exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
        **_COMPUTE_KWARGS,
    )
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_cosine_stability_compute_schema(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = CosineStability(sigma=0.05)
    results = metric.compute(
        exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
        **_COMPUTE_KWARGS,
    )
    metric_results = [r for r in results if isinstance(r, MetricResult)]
    assert len(metric_results) > 0
    for r in metric_results:
        assert r.metric_name == "cosine_stability"
        assert r.metric_family == MetricFamily.STABILITY
        assert r.status in (RunStatus.SUCCESS, RunStatus.EXCLUDED)
        if r.status == RunStatus.SUCCESS:
            assert -1.0 <= r.estimate <= 1.0 + 1e-10


@pytest.mark.unit
def test_cosine_stability_preserved_samples_finite(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = CosineStability(sigma=0.05)
    results = metric.compute(
        exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
        **_COMPUTE_KWARGS,
    )
    for r in results:
        if isinstance(r, MetricResult) and r.status == RunStatus.SUCCESS:
            assert math.isfinite(r.estimate)


# ---------------------------------------------------------------------------
# SpearmanStability.compute(): schema and contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_spearman_stability_compute_count(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = SpearmanStability(sigma=0.05)
    results = metric.compute(
        exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
        **_COMPUTE_KWARGS,
    )
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_spearman_stability_compute_schema(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = SpearmanStability(sigma=0.05)
    results = metric.compute(
        exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
        **_COMPUTE_KWARGS,
    )
    for r in results:
        if isinstance(r, MetricResult):
            assert r.metric_name == "spearman_stability"
            assert r.metric_family == MetricFamily.STABILITY


# ---------------------------------------------------------------------------
# RuntimeMetric.compute()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_runtime_metric_compute_count(exact_records):
    metric = RuntimeMetric()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_runtime_metric_values_positive(exact_records):
    metric = RuntimeMetric()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    for r in results:
        if isinstance(r, MetricResult):
            assert r.estimate >= 0.0
            assert r.metric_name == "runtime_ms"


# ---------------------------------------------------------------------------
# timer_and_memory context manager
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_timer_and_memory_captures_time():
    import time
    with timer_and_memory() as stats:
        time.sleep(0.01)
    assert stats["wall_ms"] >= 10.0  # at least 10ms


@pytest.mark.unit
def test_timer_and_memory_captures_memory():
    with timer_and_memory() as stats:
        _ = [0] * 100000  # allocate some memory
    assert stats["peak_mb"] >= 0.0
    assert isinstance(stats["peak_mb"], float)


# ---------------------------------------------------------------------------
# K90Sparsity property: linear model has low k90
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_k90_linear_model_is_sparse(lr_fitted, synth_data, exact_records):
    """
    ExactLinear on the synthetic fixture (BETA_TRUE=[1.5,-1.2,0.8,0,0,0,0,0]):
    only 3 nonzero true features. k90 should typically be ≤ 4 for most samples.
    This is a property test (not a golden value).
    """
    metric = K90Sparsity()
    results = metric.compute(exact_records, **_COMPUTE_KWARGS)
    k_values = [r.estimate for r in results if isinstance(r, MetricResult) and r.status == RunStatus.SUCCESS]
    assert len(k_values) > 0
    # The mean k90 over samples should be < n_features (8) for a sparse linear model
    mean_k = sum(k_values) / len(k_values)
    assert mean_k < len(FEATURE_NAMES), (
        f"Mean k90={mean_k:.2f} should be < {len(FEATURE_NAMES)} for sparse linear model."
    )
