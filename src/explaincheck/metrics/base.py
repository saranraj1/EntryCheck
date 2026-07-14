"""
ExplainCheck — abstract base class for all metric implementations.

Every metric must declare:
  - direction: "higher_is_better" or "lower_is_better"
  - range: (min, max) or (None, None) if unbounded
  - requires_prediction_preservation: bool
  - assumptions: list of strings
  - parameters: dict of parameter names and values
  - aggregation_method: str

Every metric must implement compute() returning MetricResult objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from explaincheck.contracts import (
    AttributionRecord,
    FailureRecord,
    MetricFamily,
    MetricResult,
    ModelFamily,
)


class BaseMetric(ABC):
    """Abstract interface for all metric implementations."""

    family: MetricFamily
    name: str
    direction: str            # "higher_is_better" or "lower_is_better"
    value_range: tuple[float | None, float | None] = (None, None)
    requires_prediction_preservation: bool = False
    aggregation_method: str = "mean"

    @property
    @abstractmethod
    def assumptions(self) -> list[str]:
        """Human-readable list of assumptions this metric makes."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """All parameters that affect the metric value."""

    @abstractmethod
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
        """
        Compute the metric over a list of AttributionRecord objects.

        Returns one MetricResult per sample (or per stressor level).
        Failures are returned as FailureRecord — never silently discarded.
        """

    def validate_attributions(
        self, attributions: list[AttributionRecord]
    ) -> None:
        """
        Raise ValueError if attributions are invalid for this metric.
        Called at the start of compute() by default.
        """
        if not attributions:
            raise ValueError(f"{self.name}: received empty attribution list.")
        if self.requires_prediction_preservation:
            # Subclasses must filter for prediction-preserved pairs.
            pass
