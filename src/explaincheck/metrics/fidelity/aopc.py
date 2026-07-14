"""
ExplainCheck — Deletion Fidelity AOPC metric.

Migrated from Phase 0 (run_phase0.py) without changing scientific definitions.

Scientific definition (frozen from Phase 0):
    For each sample i and k=1..kmax, replace the k highest-ranked features
    (by absolute attribution) with the training-set baseline (mean).
    Compute |logit_original - logit_after_replacement| at each step.
    AOPC = mean over all k steps and all samples.

Replacement: training-set mean (data-aware, as specified in protocol-v1.yaml).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from explaincheck.contracts import (
    AttributionRecord,
    FailureRecord,
    MetricFamily,
    MetricResult,
    ModelFamily,
    PredictionPreservationStatus,
    RunStatus,
)
from explaincheck.metrics.base import BaseMetric
from explaincheck.provenance import utc_now_iso


def _topk_idx(row: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k largest-absolute-value features. Stable sort (matches Phase 0)."""
    return np.argsort(-np.abs(row), kind="stable")[:k]


def deletion_fidelity_aopc_single(
    x: np.ndarray,
    attribution: np.ndarray,
    weights: np.ndarray,
    bias: float,
    baseline: np.ndarray,
    kmax: int,
) -> float:
    """
    Compute deletion fidelity AOPC for one sample.

    Migrated directly from Phase 0: cumulative mean of |logit_delta| after
    sequentially masking top-ranked features to baseline.
    """
    base_logit = float(x @ weights + bias)
    order = _topk_idx(attribution, kmax)
    drops: list[float] = []
    xm = x.copy()
    for j in order:
        xm[j] = baseline[j]
        drops.append(abs(base_logit - float(xm @ weights + bias)))
    return float(np.mean(drops))


class DeletionFidelityAOPC(BaseMetric):
    """
    Mean cumulative absolute logit change after sequential feature deletion.

    Direction: higher = better fidelity (attribution ranks features that
    matter most to the model's logit).
    """

    family = MetricFamily.FIDELITY
    name = "deletion_fidelity_aopc"
    direction = "higher_is_better"
    value_range = (0.0, None)
    requires_prediction_preservation = False
    aggregation_method = "mean"

    def __init__(self, kmax: int = 3) -> None:
        self.kmax = kmax

    @property
    def assumptions(self) -> list[str]:
        return [
            "Model logit is well-defined and accessible.",
            "Replacement value is the training-set mean (data-aware).",
            "Features are replaced in descending order of absolute attribution.",
            "Phase 0 definition: cumulative mean over k=1..kmax steps.",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "kmax": self.kmax,
            "replacement": "training_set_mean",
            "ranking": "abs_attribution_descending",
            "aggregation": "mean_over_k_and_samples",
        }

    def compute(
        self,
        attributions: list[AttributionRecord],
        *,
        run_id: str,
        protocol_version: str,
        dataset: str,
        dataset_version: str,
        split_hash: str,
        model_family: ModelFamily,
        model_hash: str,
        seed: int,
        weights: np.ndarray,
        bias: float,
        baseline: np.ndarray,
        X: np.ndarray,
        stressor: str | None = None,
        stress_level: str | None = None,
        subgroup: str | None = None,
        subgroup_value: str | None = None,
        **kwargs: Any,
    ) -> list[MetricResult | FailureRecord]:
        self.validate_attributions(attributions)
        results: list[MetricResult | FailureRecord] = []

        for i, rec in enumerate(attributions):
            t0 = time.perf_counter()
            try:
                attr = np.array(rec.attribution)
                if any(not math.isfinite(v) for v in rec.attribution):
                    raise ValueError("Attribution contains NaN or Inf.")
                score = deletion_fidelity_aopc_single(
                    X[i], attr, weights, bias, baseline, self.kmax
                )
                rt = (time.perf_counter() - t0) * 1000
                results.append(
                    MetricResult(
                        run_id=run_id,
                        protocol_version=protocol_version,
                        dataset=dataset,
                        dataset_version=dataset_version,
                        split_hash=split_hash,
                        model_family=model_family,
                        model_hash=model_hash,
                        explainer=rec.explainer,
                        explainer_version=rec.explainer_version,
                        seed=seed,
                        sample_id=rec.sample_id,
                        metric_family=self.family,
                        metric_name=self.name,
                        metric_k=self.kmax,
                        stressor=stressor,
                        stress_level=stress_level,
                        subgroup=subgroup,
                        subgroup_value=subgroup_value,
                        prediction_preservation_status=PredictionPreservationStatus.NOT_APPLICABLE,
                        estimate=score,
                        runtime_ms=rt,
                        status=RunStatus.SUCCESS,
                    )
                )
            except Exception as exc:
                rt = (time.perf_counter() - t0) * 1000
                results.append(
                    FailureRecord(
                        run_id=run_id,
                        timestamp=utc_now_iso(),
                        dataset=dataset,
                        model_family=model_family,
                        explainer=rec.explainer,
                        metric_name=self.name,
                        seed=seed,
                        failure_reason=str(exc),
                        is_deterministic=True,
                        excluded=False,
                    )
                )
        return results
