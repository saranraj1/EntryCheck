"""
ExplainCheck — abstract base class for all metric implementations (Option B+, DR-006A §1).

Every metric must declare:
  - family:    MetricFamily enum value
  - name:      str
  - direction: "higher_is_better" or "lower_is_better"
  - value_range, requires_prediction_preservation, aggregation_method

Every metric must implement compute(context: ContextT) returning MetricResult objects.

The Generic[ContextT] parameter removes the need for # type: ignore[override] on
subclass compute() methods — each subclass declares its specific context type,
and mypy verifies that callers pass the correct context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from explaincheck.contracts import (
    FailureRecord,
    MetricFamily,
    MetricResult,
)
from explaincheck.metrics.contexts import BaseMetricContext

ContextT = TypeVar("ContextT", bound=BaseMetricContext)


class BaseMetric(ABC, Generic[ContextT]):
    """
    Abstract interface for all metric implementations.

    Type parameter ContextT is the specific BaseMetricContext subclass
    that this metric requires.  Callers must pass the correct context type;
    mypy enforces this without any type: ignore suppression.
    """

    family: MetricFamily
    name: str
    direction: str  # "higher_is_better" or "lower_is_better"
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
    def compute(self, context: ContextT) -> list[MetricResult | FailureRecord]:
        """
        Compute the metric using the typed context.

        Returns one MetricResult per sample (or per stressor level).
        Failures are returned as FailureRecord — never silently discarded.

        The context carries all provenance fields and scientific inputs.
        Subclasses declare their specific ContextT and mypy verifies compatibility.
        """

    def validate_context(self, context: BaseMetricContext) -> None:
        """
        Raise ValueError if the context is invalid for this metric.
        Called at the start of compute() by default.

        Context-level validation (non-empty attributions, valid shapes) is
        already performed by Pydantic at context construction time.
        Metric-specific cross-field validation goes here.
        """
        if self.requires_prediction_preservation:
            pass

    def validate_attributions(self, attributions: list) -> None:  # noqa: ANN401
        """
        Legacy validation helper for subclasses that pass attributions directly.
        Kept for backward compatibility with Stage 4 metrics pending migration.
        Raises ValueError if attributions list is empty.
        """
        if not attributions:
            raise ValueError(f"{self.name}: received empty attribution list.")
