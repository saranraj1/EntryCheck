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
from typing import Any as _Any

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
        k90 value. 0 if all attributions are zero.

    Raises
    ------
    ValueError
        If attribution contains NaN/Inf or is empty.
    """
    if not np.all(np.isfinite(attribution)):
        raise ValueError("Attribution contains NaN or Inf.")
    if len(attribution) == 0:
        raise ValueError("Empty attribution vector.")
    abs_a = np.abs(attribution)
    total = float(abs_a.sum())
    if total == 0.0:
        return 0
    sorted_desc = np.sort(abs_a)[::-1]
    cumsum = np.cumsum(sorted_desc)
    passing = np.where(cumsum / total >= threshold)[0]
    if len(passing) == 0:
        return int(len(attribution))
    return int(passing[0]) + 1  # 1-indexed count


class K90Sparsity(BaseMetric[_Any]):
    """
    k90 Sparsity metric.

    Direction: lower = sparser explanations (fewer features needed to capture 90% of attribution mass).
    """

    family = MetricFamily.SPARSITY
    name = "k90_sparsity"
    direction = "lower_is_better"
    value_range = (0.0, None)
    requires_prediction_preservation = False
    aggregation_method = "mean"

    def __init__(self, *, threshold: float = 0.90) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self._threshold = threshold

    @property
    def assumptions(self) -> list[str]:
        return [
            "Attribution vectors are dense and finite.",
            "k90 is computed over absolute values (direction-agnostic sparsity).",
            "Zero attribution vector → k90 = 0 (trivially sparse).",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {"threshold": self._threshold}

    def compute(  # type: ignore[override]  # Stage 4 quarantine: pending StabilityContext migration
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
        stressor: str | None = None,
        stress_level: str | None = None,
        subgroup: str | None = None,
        subgroup_value: str | None = None,
        **kwargs: Any,
    ) -> list[MetricResult | FailureRecord]:
        self.validate_attributions(attributions)
        results: list[MetricResult | FailureRecord] = []

        for rec in attributions:
            t0 = time.perf_counter()
            try:
                a = np.array(rec.attribution, dtype=float)
                k = k90_sparsity(a, threshold=self._threshold)
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
