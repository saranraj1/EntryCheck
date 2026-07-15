"""
ExplainCheck — Runtime measurement utilities (Stage 4, DR-003A).

Provides:
  - timer_and_memory(): context manager for wall-clock time and peak memory.
  - RuntimeMetric: extracts runtime_ms from AttributionRecord.

These are infrastructure metrics, not scientific correctness metrics.
They capture explainer scalability for the benchmark's operational evaluation dimension.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

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


@contextmanager
def timer_and_memory() -> Generator[dict[str, float], None, None]:
    """
    Context manager that measures wall-clock time and peak Python memory allocation.

    Usage:
        with timer_and_memory() as stats:
            do_work()
        print(stats["wall_ms"], stats["peak_mb"])
    """
    result: dict[str, float] = {"wall_ms": 0.0, "peak_mb": 0.0}
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["wall_ms"] = round(elapsed, 3)
        result["peak_mb"] = round(peak / 1024.0 / 1024.0, 4)


class RuntimeMetric(BaseMetric):
    """
    Explanation runtime (wall-clock ms) extracted from AttributionRecord.runtime_ms.

    Direction: lower = faster.
    """

    family = MetricFamily.RUNTIME
    name = "runtime_ms"
    direction = "lower_is_better"
    value_range = (0.0, None)
    requires_prediction_preservation = False
    aggregation_method = "mean"

    @property
    def assumptions(self) -> list[str]:
        return [
            "runtime_ms was recorded by the explainer adapter during explain().",
            "Wall-clock time includes SHAP/LIME computation, not Python overhead from this metric.",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

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
        stressor: str | None = None,
        stress_level: str | None = None,
        subgroup: str | None = None,
        subgroup_value: str | None = None,
        **kwargs: Any,
    ) -> list[MetricResult | FailureRecord]:
        self.validate_attributions(attributions)
        results: list[MetricResult | FailureRecord] = []
        ts = utc_now_iso()

        for rec in attributions:
            t0 = time.perf_counter()
            try:
                rt_val = rec.runtime_ms
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
                        estimate=float(rt_val),
                        runtime_ms=rt,
                        status=RunStatus.SUCCESS,
                    )
                )
            except Exception as exc:
                results.append(
                    FailureRecord(
                        run_id=run_id,
                        timestamp=ts,
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
