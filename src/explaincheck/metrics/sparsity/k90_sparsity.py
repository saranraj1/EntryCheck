"""
ExplainCheck — k90 Sparsity metric (Stage 4, DR-003A).

Scientific definition:
    k90 = minimum number of top-|attribution| features needed to account
    for 90% of the total L1 attribution mass.

    k90 = min{k : sum(|A|_sorted_desc[:k]) / sum(|A|) >= 0.90}

    - Attribution vector is sorted by absolute value descending.
    - k90 = 1 if a single feature accounts for >= 90% of total mass.
    - k90 = p if the condition is never met (uniform attributions).
    - If total |A| mass == 0, k90 = 0 (zero attribution vector — trivially sparse).

Direction: lower = sparser (fewer features account for most attribution mass).
Range: [0, n_features].
Prediction preservation: not required (sparsity is a property of a single explanation).
"""

from __future__ import annotations

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
from explaincheck.metrics.contexts import SparsityContext
from explaincheck.provenance import utc_now_iso


def k90_sparsity(attribution: np.ndarray, threshold: float = 0.90) -> int:
    """
    Compute k90 sparsity: minimum features to account for `threshold` of L1 mass.

    Parameters
    ----------
    attribution : np.ndarray
        Dense attribution vector (1D).
    threshold : float
        Mass coverage threshold. Default 0.90 (90%).

    Returns
    -------
    int
        k90 value.  0 if total absolute mass is zero.

    Raises
    ------
    ValueError
        If the array contains NaN or Inf values.
    """
    if not np.all(np.isfinite(attribution)):
        raise ValueError("Attribution contains NaN or Inf.")
    total_mass = float(np.sum(np.abs(attribution)))
    if total_mass == 0.0:
        return 0
    sorted_abs = np.sort(np.abs(attribution))[::-1]
    cumulative = np.cumsum(sorted_abs)
    k90_indices = np.where(cumulative / total_mass >= threshold)[0]
    if len(k90_indices) == 0:
        return len(attribution)
    return int(k90_indices[0]) + 1


class K90Sparsity(BaseMetric[SparsityContext]):
    """
    K90 sparsity metric.

    Minimum number of top-|attribution| features needed to cover 90% of the
    total L1 attribution mass.  Lower is sparser (more concentrated attribution).
    Migrated to Option B+ typed context interface (DR-008).
    """

    family = MetricFamily.SPARSITY
    name = "k90_sparsity"
    direction = "lower_is_better"
    value_range = (0.0, None)
    requires_prediction_preservation = False
    aggregation_method = "mean"

    def __init__(self, *, threshold: float = 0.90) -> None:
        if not np.isfinite(threshold) or not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be a finite value in (0, 1], got {threshold}")
        self._threshold = threshold

    @property
    def assumptions(self) -> list[str]:
        return [
            "Attribution vectors are dense and finite.",
            "k90 is computed over absolute values (direction-agnostic sparsity).",
            "Zero attribution vector -> k90 = 0 (trivially sparse).",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self._threshold}

    def compute(
        self,
        context: SparsityContext,
    ) -> list[MetricResult | FailureRecord]:
        """
        Compute k90 sparsity for each AttributionRecord in context.attributions.

        Uses context.threshold if provided; otherwise falls back to self._threshold.
        The threshold in context is validated at construction time by SparsityContext.
        """
        attributions: tuple[AttributionRecord, ...] = context.attributions
        run_id = context.run_id
        protocol_version = context.protocol_version
        dataset = context.dataset
        dataset_version = context.dataset_version
        split_hash = context.split_hash
        model_family = ModelFamily(context.model_family)
        model_hash = context.model_hash
        seed = context.seed
        stressor = context.stressor
        stress_level = context.stress_level
        subgroup = context.subgroup
        subgroup_value = context.subgroup_value
        threshold = context.threshold

        results: list[MetricResult | FailureRecord] = []

        for rec in attributions:
            t0 = time.perf_counter()
            try:
                a = np.array(rec.attribution, dtype=float)
                k = k90_sparsity(a, threshold=threshold)
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
                        stressor=stressor,
                        stress_level=stress_level,
                        subgroup=subgroup,
                        subgroup_value=subgroup_value,
                        prediction_preservation_status=PredictionPreservationStatus.NOT_APPLICABLE,
                        estimate=float(k),
                        runtime_ms=rt,
                        status=RunStatus.SUCCESS,
                    )
                )
            except Exception as exc:
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
