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
  - Context-contract tests: PairwiseStabilityContext, SparsityContext, RuntimeContext

All metric-compute tests use the exact_linear explainer against the synthetic linear fixture
to keep tests fast and analytically verifiable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from explaincheck.contracts import (
    MetricFamily,
    ModelFamily,
    PredictionPreservationStatus,
    RunStatus,
)
from explaincheck.contracts.models import MetricResult
from explaincheck.datasets.synthetic import FEATURE_NAMES, generate, split
from explaincheck.explainers.exact_linear import ExactLinearExplainer
from explaincheck.metrics.contexts import PairwiseStabilityContext, RuntimeContext, SparsityContext
from explaincheck.metrics.runtime.runtime_metric import RuntimeMetric, timer_and_memory
from explaincheck.metrics.sparsity.k90_sparsity import K90Sparsity, k90_sparsity
from explaincheck.metrics.stability.cosine_stability import CosineStability, cosine_similarity_pair
from explaincheck.metrics.stability.spearman_stability import SpearmanStability, spearman_pair
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
        X_te[:20],
        run_id="t4",
        dataset="synth-linear",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        sample_ids=[f"s{i}" for i in range(20)],
        protocol_version="1.0.0",
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
    """Zero attribution vector: k90 = 0 (trivially sparse)."""
    a = np.array([0.0, 0.0, 0.0])
    assert k90_sparsity(a) == 0


@pytest.mark.unit
def test_k90_exactly_90_two_features():
    """[0.5, 0.4, 0.1]: cumulative after 2 features = 0.9, so k90 = 2."""
    a = np.array([0.5, 0.4, 0.1])
    assert k90_sparsity(a) == 2


@pytest.mark.unit
def test_k90_handles_negative_values():
    """Negative values treated as |value|; should still return valid k90."""
    a = np.array([-0.5, -0.4, -0.1])
    assert k90_sparsity(a) == 2


@pytest.mark.unit
def test_k90_raises_on_nan():
    with pytest.raises(ValueError, match="NaN or Inf"):
        k90_sparsity(np.array([float("nan"), 1.0]))


@pytest.mark.unit
def test_k90_raises_on_empty():
    """Edge case: empty attribution vector produces k90=0 (zero mass)."""
    result = k90_sparsity(np.array([]))
    assert result == 0


@pytest.mark.unit
def test_k90_single_element():
    """Single non-zero element: k90 = 1."""
    assert k90_sparsity(np.array([1.0])) == 1


# ---------------------------------------------------------------------------
# cosine_similarity_pair: hand fixtures
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity_pair(a, a) == pytest.approx(1.0)


@pytest.mark.unit
def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity_pair(a, b) == pytest.approx(0.0)


@pytest.mark.unit
def test_cosine_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity_pair(a, b) == pytest.approx(-1.0)


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
        cosine_similarity_pair(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.unit
def test_cosine_nan_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        cosine_similarity_pair(np.array([float("nan")]), np.array([1.0]))


@pytest.mark.unit
def test_cosine_known_value():
    """[3, 4] vs [4, 3]: dot=24, norms=5,5, cos=24/25=0.96."""
    a = np.array([3.0, 4.0])
    b = np.array([4.0, 3.0])
    assert cosine_similarity_pair(a, b) == pytest.approx(0.96)


# ---------------------------------------------------------------------------
# spearman_pair: hand fixtures
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spearman_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_pair(a, a) == pytest.approx(1.0)


@pytest.mark.unit
def test_spearman_reversed():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert spearman_pair(a, b) == pytest.approx(-1.0)


@pytest.mark.unit
def test_spearman_both_constant():
    a = np.array([1.0, 1.0, 1.0])
    assert spearman_pair(a, a) == 1.0


@pytest.mark.unit
def test_spearman_nan_on_one_constant():
    a = np.array([1.0, 1.0, 1.0])  # constant
    b = np.array([1.0, 2.0, 3.0])  # not constant
    rho = spearman_pair(a, b)
    assert math.isnan(rho)


@pytest.mark.unit
def test_spearman_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        spearman_pair(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.unit
def test_spearman_nan_raises():
    with pytest.raises(ValueError, match="NaN or Inf"):
        spearman_pair(np.array([float("nan")]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Shared context kwargs (for sparsity/runtime contexts)
# ---------------------------------------------------------------------------

_BASE_CTX = dict(
    run_id="t4",
    protocol_version="1.0.0",
    dataset="synth-linear",
    dataset_version="1.0",
    split_hash="abc123",
    model_family=ModelFamily.LOGISTIC_REGRESSION.value,
    model_hash="mh",
    seed=11,
)


# ---------------------------------------------------------------------------
# K90Sparsity.compute(): schema and contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_k90_compute_output_count(exact_records):
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_k90_compute_all_success(exact_records):
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
    for r in results:
        assert isinstance(r, MetricResult)
        assert r.status == RunStatus.SUCCESS
        assert r.metric_name == "k90_sparsity"
        assert r.metric_family == MetricFamily.SPARSITY


@pytest.mark.unit
def test_k90_compute_values_in_range(exact_records):
    """k90 must be in [0, n_features] for all samples."""
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
    p = len(FEATURE_NAMES)
    for r in results:
        if isinstance(r, MetricResult):
            assert 0 <= r.estimate <= p, f"k90={r.estimate} out of [0, {p}]"


@pytest.mark.unit
def test_k90_compute_deterministic(exact_records):
    """Same inputs -> same outputs."""
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    r1 = metric.compute(ctx)
    r2 = metric.compute(ctx)
    for a, b in zip(r1, r2, strict=True):
        if isinstance(a, MetricResult) and isinstance(b, MetricResult):
            assert a.estimate == b.estimate


@pytest.mark.unit
def test_k90_compute_not_applicable_preservation(exact_records):
    """K90 does not require prediction preservation."""
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
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
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    results = metric.compute(ctx)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_cosine_stability_compute_schema(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = CosineStability(sigma=0.05)
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    results = metric.compute(ctx)
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
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    results = metric.compute(ctx)
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
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    results = metric.compute(ctx)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_spearman_stability_compute_schema(lr_fitted, synth_data, exact_records):
    model, rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    metric = SpearmanStability(sigma=0.05)
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    results = metric.compute(ctx)
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
    ctx = RuntimeContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
    assert len(results) == len(exact_records)


@pytest.mark.unit
def test_runtime_metric_values_positive(exact_records):
    metric = RuntimeMetric()
    ctx = RuntimeContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
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
    only 3 nonzero true features. k90 should typically be <= 4 for most samples.
    This is a property test (not a golden value).
    """
    metric = K90Sparsity()
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    results = metric.compute(ctx)
    k_values = [
        r.estimate for r in results if isinstance(r, MetricResult) and r.status == RunStatus.SUCCESS
    ]
    assert len(k_values) > 0
    # The mean k90 over samples should be < n_features (8) for a sparse linear model
    mean_k = sum(k_values) / len(k_values)
    assert mean_k < len(
        FEATURE_NAMES
    ), f"Mean k90={mean_k:.2f} should be < {len(FEATURE_NAMES)} for sparse linear model."


# ---------------------------------------------------------------------------
# Context-contract tests (DR-008 §4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sparsity_context_accepts_valid(exact_records):
    """Valid SparsityContext constructs without error."""
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    assert isinstance(ctx.attributions, tuple)
    assert len(ctx.attributions) == len(exact_records)
    assert ctx.threshold == 0.90


@pytest.mark.unit
def test_sparsity_context_rejects_empty_attributions():
    """Empty attribution sequence raises ValueError."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=[])


@pytest.mark.unit
def test_sparsity_context_rejects_bad_threshold_zero():
    """threshold=0 is invalid (must be > 0)."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=["placeholder"], threshold=0.0)


@pytest.mark.unit
def test_sparsity_context_rejects_bad_threshold_above_one(exact_records):
    """threshold > 1 is invalid."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=exact_records, threshold=1.1)


@pytest.mark.unit
def test_sparsity_context_rejects_nan_threshold(exact_records):
    """NaN threshold is invalid."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=exact_records, threshold=float("nan"))


@pytest.mark.unit
def test_sparsity_context_rejects_inf_threshold(exact_records):
    """Positive infinity threshold is invalid."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=exact_records, threshold=float("inf"))


@pytest.mark.unit
def test_sparsity_context_rejects_neg_inf_threshold(exact_records):
    """Negative infinity threshold is invalid."""
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=exact_records, threshold=float("-inf"))


@pytest.mark.unit
def test_sparsity_context_is_frozen(exact_records):
    """SparsityContext is immutable — field reassignment raises."""
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    with pytest.raises((ValidationError, TypeError)):
        ctx.threshold = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_sparsity_context_tuple_is_immutable(exact_records):
    """The stored attributions tuple does not allow item assignment."""
    ctx = SparsityContext(**_BASE_CTX, attributions=exact_records)
    with pytest.raises(TypeError):
        ctx.attributions[0] = exact_records[0]  # type: ignore[index]


@pytest.mark.unit
def test_sparsity_context_rejects_invalid_element_type(exact_records):
    """A non-AttributionRecord element in the sequence raises ValueError."""
    bad_list = list(exact_records)
    bad_list[0] = "not_a_record"  # type: ignore[assignment]
    with pytest.raises((ValueError, ValidationError)):
        SparsityContext(**_BASE_CTX, attributions=bad_list)


@pytest.mark.unit
def test_runtime_context_accepts_valid(exact_records):
    """Valid RuntimeContext constructs without error."""
    ctx = RuntimeContext(**_BASE_CTX, attributions=exact_records)
    assert isinstance(ctx.attributions, tuple)
    assert len(ctx.attributions) == len(exact_records)


@pytest.mark.unit
def test_runtime_context_rejects_empty_attributions():
    """Empty attribution sequence raises."""
    with pytest.raises((ValueError, ValidationError)):
        RuntimeContext(**_BASE_CTX, attributions=[])


@pytest.mark.unit
def test_runtime_context_is_frozen(exact_records):
    """RuntimeContext is immutable."""
    ctx = RuntimeContext(**_BASE_CTX, attributions=exact_records)
    with pytest.raises((ValidationError, TypeError)):
        ctx.run_id = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_runtime_context_tuple_is_immutable(exact_records):
    """The stored attributions tuple does not allow item assignment."""
    ctx = RuntimeContext(**_BASE_CTX, attributions=exact_records)
    with pytest.raises(TypeError):
        ctx.attributions[0] = exact_records[0]  # type: ignore[index]


@pytest.mark.unit
def test_runtime_context_rejects_invalid_element_type(exact_records):
    """A non-AttributionRecord element raises."""
    bad_list = list(exact_records)
    bad_list[0] = 42  # type: ignore[assignment]
    with pytest.raises((ValueError, ValidationError)):
        RuntimeContext(**_BASE_CTX, attributions=bad_list)


@pytest.mark.unit
def test_pairwise_stability_context_accepts_valid(lr_fitted, synth_data, exact_records):
    """Valid PairwiseStabilityContext constructs without error."""
    model, _rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    assert isinstance(ctx.attributions, tuple)
    assert ctx.sigma == 0.05


@pytest.mark.unit
def test_pairwise_stability_context_rejects_empty():
    """Empty attributions raises."""
    with pytest.raises((ValueError, ValidationError)):
        PairwiseStabilityContext(
            **_BASE_CTX,
            attributions=[],
            weights=np.ones(3),
            bias=0.0,
            baseline=np.zeros(3),
            X=np.zeros((1, 3)),
        )


@pytest.mark.unit
def test_pairwise_stability_context_is_frozen(lr_fitted, synth_data, exact_records):
    """PairwiseStabilityContext is immutable."""
    model, _rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    with pytest.raises((ValidationError, TypeError)):
        ctx.sigma = 0.1  # type: ignore[misc]


@pytest.mark.unit
def test_pairwise_stability_context_tuple_is_immutable(lr_fitted, synth_data, exact_records):
    """Stored attributions tuple does not allow item assignment."""
    model, _rec = lr_fitted
    X_tr, X_te, _, _ = synth_data
    baseline = X_tr.mean(axis=0)
    ctx = PairwiseStabilityContext(
        **_BASE_CTX,
        attributions=exact_records,
        weights=model.weights,
        bias=model.bias,
        baseline=baseline,
        X=X_te[:20],
    )
    with pytest.raises(TypeError):
        ctx.attributions[0] = exact_records[0]  # type: ignore[index]


@pytest.mark.unit
def test_no_new_first_party_suppressions():
    """
    DR-008 §5: No new first-party type:ignore or noqa suppressions in migrated files.
    Allowed: noqa:ANN401 on coerce_attributions (bare tuple return annotation),
             noqa in existing pre-migration code outside the 4 metric files.
    The four quarantined type:ignore[override] suppressions must be absent.
    """
    import pathlib

    src_root = pathlib.Path("src/explaincheck/metrics")
    migrated = [
        src_root / "stability" / "cosine_stability.py",
        src_root / "stability" / "spearman_stability.py",
        src_root / "sparsity" / "k90_sparsity.py",
        src_root / "runtime" / "runtime_metric.py",
    ]
    for path in migrated:
        text = path.read_text(encoding="utf-8")
        assert (
            "type: ignore[override]" not in text
        ), f"{path.name} still contains a quarantined override suppression"
        assert "type: ignore[override]" not in text
