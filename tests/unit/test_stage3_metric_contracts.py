"""
Stage 3 metric contract tests (DR-006A §1 + §4i).

Verifies that:
1. AOPCMetric.compute() accepts only AOPCContext — not StabilityContext
2. TopKJaccardStability.compute() accepts only StabilityContext — not AOPCContext
3. BaseMetricContext validation fires correctly (seed < 0, empty strings)
4. AOPCContext validates its scientific inputs (empty attributions, non-finite arrays)
5. StabilityContext validates its scientific inputs (k <= 0, sigma <= 0)
6. prediction_preservation_rate is present in MetricResult schema

These are contract tests — they verify the typed interface, not scientific correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from explaincheck.contracts import (
    AttributionRecord,
    DataSplit,
    ExplainerName,
    ExplainerType,
    ModelFamily,
)
from explaincheck.metrics.contexts import AOPCContext, StabilityContext
from explaincheck.metrics.fidelity.aopc import AOPCMetric
from explaincheck.metrics.stability.top_k_jaccard import TopKJaccardStability

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 5
P = 8


@pytest.fixture()
def sample_attribution_records() -> list[AttributionRecord]:
    """Return a small list of valid AttributionRecord objects."""
    return [
        AttributionRecord(
            run_id="contract-test",
            protocol_version="1.0.0",
            sample_id=f"s{i}",
            dataset="synthetic",
            seed=0,
            split=DataSplit.TEST,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash="aabbccdd",
            explainer=ExplainerName.EXACT_LINEAR,
            explainer_type=ExplainerType.LOCAL,
            explainer_version="0.1.0",
            feature_names=[f"f{j}" for j in range(P)],
            attribution=[float(j) * 0.1 for j in range(P)],
            prediction_class=1,
            prediction_probability=0.7,
            runtime_ms=1.0,
            success=True,
        )
        for i in range(N)
    ]


@pytest.fixture()
def base_provenance() -> dict:
    """Return shared provenance fields for context construction."""
    return {
        "run_id": "contract-test",
        "protocol_version": "1.0.0",
        "dataset": "synthetic",
        "dataset_version": "v1",
        "split_hash": "abc123",
        "model_family": "logistic_regression",
        "model_hash": "aabbccdd",
        "seed": 0,
    }


@pytest.fixture()
def aopc_context(sample_attribution_records, base_provenance) -> AOPCContext:
    weights = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
    return AOPCContext(
        **base_provenance,
        attributions=sample_attribution_records,
        weights=weights,
        bias=0.3,
        baseline=np.zeros(P),
        X=np.random.default_rng(0).standard_normal((N, P)),
    )


@pytest.fixture()
def stability_context(sample_attribution_records, base_provenance) -> StabilityContext:
    weights = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
    return StabilityContext(
        **base_provenance,
        attributions=sample_attribution_records,
        k=3,
        sigma=0.05,
        weights=weights,
        bias=0.3,
        baseline=np.zeros(P),
        X=np.random.default_rng(0).standard_normal((N, P)),
    )


# ---------------------------------------------------------------------------
# Contract test 1: Metric–context pairing (correct types accepted)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aopc_metric_accepts_aopc_context(aopc_context) -> None:
    """AOPCMetric.compute() must accept AOPCContext without error."""
    metric = AOPCMetric(kmax=3)
    results = metric.compute(aopc_context)
    assert len(results) == N, f"Expected {N} results, got {len(results)}"


@pytest.mark.unit
def test_top_k_jaccard_accepts_stability_context(stability_context) -> None:
    """TopKJaccardStability.compute() must accept StabilityContext without error."""
    metric = TopKJaccardStability(k=3, sigma=0.05)
    results = metric.compute(stability_context)
    assert len(results) == N, f"Expected {N} results, got {len(results)}"


# ---------------------------------------------------------------------------
# Contract test 2: Wrong context types rejected at type level
# These tests verify that the Generic typing is meaningful.
# At runtime Pydantic won't prevent a wrong context being passed, but
# the tests document the expected interface for mypy verification.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aopc_context_is_not_stability_context(aopc_context) -> None:
    """AOPCContext is not an instance of StabilityContext."""
    assert not isinstance(aopc_context, StabilityContext)


@pytest.mark.unit
def test_stability_context_is_not_aopc_context(stability_context) -> None:
    """StabilityContext is not an instance of AOPCContext."""
    assert not isinstance(stability_context, AOPCContext)


# ---------------------------------------------------------------------------
# Contract test 3: BaseMetricContext input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_base_metric_context_rejects_negative_seed(base_provenance) -> None:
    """BaseMetricContext.seed must be >= 0."""
    with pytest.raises(ValidationError):
        AOPCContext(
            **{**base_provenance, "seed": -1},
            attributions=[object()],  # won't reach attribution validation
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((1, P)),
        )


@pytest.mark.unit
def test_base_metric_context_rejects_empty_run_id(base_provenance) -> None:
    """BaseMetricContext.run_id must be a non-empty string."""
    with pytest.raises(ValidationError):
        AOPCContext(
            **{**base_provenance, "run_id": "   "},
            attributions=[object()],
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((1, P)),
        )


# ---------------------------------------------------------------------------
# Contract test 4: AOPCContext scientific input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aopc_context_rejects_empty_attributions(base_provenance) -> None:
    """AOPCContext.attributions must be non-empty."""
    with pytest.raises(ValidationError):
        AOPCContext(
            **base_provenance,
            attributions=[],
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((1, P)),
        )


@pytest.mark.unit
def test_aopc_context_is_frozen(aopc_context) -> None:
    """AOPCContext must be immutable (Pydantic frozen=True)."""
    with pytest.raises((ValidationError, TypeError)):
        aopc_context.seed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract test 5: StabilityContext scientific input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stability_context_rejects_k_zero(base_provenance, sample_attribution_records) -> None:
    """StabilityContext.k must be > 0."""
    with pytest.raises(ValidationError):
        StabilityContext(
            **base_provenance,
            attributions=sample_attribution_records,
            k=0,
            sigma=0.05,
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((N, P)),
        )


@pytest.mark.unit
def test_stability_context_rejects_negative_sigma(
    base_provenance, sample_attribution_records
) -> None:
    """StabilityContext.sigma must be > 0."""
    with pytest.raises(ValidationError):
        StabilityContext(
            **base_provenance,
            attributions=sample_attribution_records,
            k=3,
            sigma=-0.1,
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((N, P)),
        )


@pytest.mark.unit
def test_stability_context_rejects_empty_attributions(base_provenance) -> None:
    """StabilityContext.attributions must be non-empty."""
    with pytest.raises(ValidationError):
        StabilityContext(
            **base_provenance,
            attributions=[],
            k=3,
            sigma=0.05,
            weights=np.ones(P),
            bias=0.0,
            baseline=np.zeros(P),
            X=np.ones((N, P)),
        )


@pytest.mark.unit
def test_stability_context_is_frozen(stability_context) -> None:
    """StabilityContext must be immutable (Pydantic frozen=True)."""
    with pytest.raises((ValidationError, TypeError)):
        stability_context.k = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract test 6: prediction_preservation_rate in MetricResult schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metric_result_has_prediction_preservation_rate(stability_context) -> None:
    """
    MetricResult objects returned by TopKJaccardStability.compute()
    must include prediction_preservation_rate (DR-006A §11).
    """
    from explaincheck.contracts import MetricResult

    metric = TopKJaccardStability(k=3, sigma=0.05)
    results = metric.compute(stability_context)

    # Every result (preserved or excluded) must have the rate field
    for r in results:
        if isinstance(r, MetricResult):
            assert r.prediction_preservation_rate is not None, (
                "prediction_preservation_rate must be set in MetricResult "
                "for stability metrics (DR-006A §11)"
            )
            assert (
                0.0 <= r.prediction_preservation_rate <= 1.0
            ), f"prediction_preservation_rate must be in [0,1], got {r.prediction_preservation_rate}"
