"""
ExplainCheck — Typed, immutable metric context models (Option B+, DR-006A §1).

Each metric defines its own context class that inherits from BaseMetricContext.
Contexts are validated at construction time via Pydantic (frozen=True).
This eliminates the need for **kwargs in BaseMetric.compute() and allows
mypy to verify that each metric receives exactly the inputs it declares.

Shared provenance fields live in BaseMetricContext.
Scientific inputs live in metric-specific subclasses.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class BaseMetricContext(BaseModel):
    """
    Provenance fields shared by all metric computations.

    All contexts are immutable (frozen=True).  Metric-specific inputs go
    in subclasses; do not add optional Any fields here.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run_id: str
    protocol_version: str
    dataset: str
    dataset_version: str
    split_hash: str
    model_family: str  # ModelFamily.value — string avoids circular import
    model_hash: str
    seed: int
    stressor: str | None = None
    stress_level: str | None = None
    subgroup: str | None = None
    subgroup_value: str | None = None

    @field_validator("seed")
    @classmethod
    def seed_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"seed must be >= 0, got {v}")
        return v

    @field_validator(
        "run_id",
        "protocol_version",
        "dataset",
        "dataset_version",
        "split_hash",
        "model_family",
        "model_hash",
        mode="before",
    )
    @classmethod
    def non_empty_string(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Context string fields must be non-empty")
        return v


class AOPCContext(BaseMetricContext):
    """
    Context for AOPC (deletion fidelity) metric.

    Scientific inputs:
        attributions: list of AttributionRecord objects (one per sample)
        weights:      model weight vector (shape [p])
        bias:         model scalar bias
        baseline:     training-set mean feature vector (shape [p])
        X:            input feature matrix (shape [n, p])
    """

    # Import inside type annotation to avoid circular imports at module level
    attributions: list  # list[AttributionRecord] — typed in compute()
    weights: np.ndarray
    bias: float
    baseline: np.ndarray
    X: np.ndarray

    @field_validator("weights", "baseline", "X", mode="before")
    @classmethod
    def must_be_ndarray(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=float)

    @field_validator("attributions")
    @classmethod
    def non_empty(cls, v: list) -> list:  # noqa: ANN401
        if not v:
            raise ValueError("AOPCContext: attributions must not be empty")
        return v


class StabilityContext(BaseMetricContext):
    """
    Context for stability metrics (Top-k Jaccard).

    Scientific inputs:
        attributions:       list of AttributionRecord objects (original, pre-perturbation)
        k:                  number of top features for Jaccard similarity
        sigma:              std dev of Gaussian perturbation applied inside compute()
        weights:            model weight vector (shape [p]) — used to compute perturbed predictions
        bias:               model scalar bias — used to compute perturbed predictions
        baseline:           training-set mean feature vector (shape [p])
        X:                  input feature matrix (shape [n, p])
        explainer_instance: optional explainer object for non-analytic explainers (default None)

    Feature ordering and prediction preservation are handled inside compute().
    """

    attributions: list  # list[AttributionRecord]
    k: int
    sigma: float = 0.05
    weights: np.ndarray
    bias: float
    baseline: np.ndarray
    X: np.ndarray
    explainer_instance: object | None = None

    @field_validator("k")
    @classmethod
    def k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"k must be > 0, got {v}")
        return v

    @field_validator("sigma")
    @classmethod
    def sigma_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"sigma must be > 0, got {v}")
        return v

    @field_validator("weights", "baseline", "X", mode="before")
    @classmethod
    def must_be_ndarray(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=float)

    @field_validator("attributions")
    @classmethod
    def non_empty(cls, v: list) -> list:  # noqa: ANN401
        if not v:
            raise ValueError("StabilityContext: attributions must not be empty")
        return v


class PairwiseStabilityContext(BaseMetricContext):
    """
    Context for pairwise stability metrics (CosineStability, SpearmanStability).

    Unlike StabilityContext this context does not carry a ``k`` field — cosine and
    Spearman rank correlation operate on the full attribution vector rather than a
    top-k feature set.

    Scientific inputs:
        attributions: immutable tuple of AttributionRecord objects (original, pre-perturbation)
        sigma:        std dev of Gaussian perturbation applied inside compute() (default 0.05)
        weights:      model weight vector (shape [p]) — used to compute perturbed predictions
        bias:         model scalar bias
        baseline:     training-set mean feature vector (shape [p])
        X:            input feature matrix (shape [n, p])

    The attribution collection is stored as an immutable tuple so that nested content
    cannot be mutated even though the Pydantic model itself is frozen (DR-008 §2).
    """

    attributions: tuple  # tuple[AttributionRecord, ...]
    sigma: float = 0.05
    weights: np.ndarray
    bias: float
    baseline: np.ndarray
    X: np.ndarray

    @field_validator("attributions", mode="before")
    @classmethod
    def coerce_attributions(cls, value: object) -> tuple:  # noqa: ANN401
        """Accept any sequence; coerce to immutable tuple and validate element types."""
        from collections.abc import Sequence as _Seq

        from explaincheck.contracts import AttributionRecord

        if not isinstance(value, _Seq) or isinstance(value, (str, bytes)):
            raise ValueError("attributions must be a sequence of AttributionRecord")
        records: tuple[object, ...] = tuple(value)
        if not records:
            raise ValueError("PairwiseStabilityContext: attributions must not be empty")
        for i, rec in enumerate(records):
            if not isinstance(rec, AttributionRecord):
                raise ValueError(
                    f"attributions[{i}] must be AttributionRecord, got {type(rec).__name__}"
                )
        return records

    @field_validator("sigma")
    @classmethod
    def sigma_positive(cls, v: float) -> float:
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"sigma must be a finite positive number, got {v}")
        return v

    @field_validator("weights", "baseline", "X", mode="before")
    @classmethod
    def must_be_ndarray(cls, v: object) -> np.ndarray:
        return np.asarray(v, dtype=float)


class SparsityContext(BaseMetricContext):
    """
    Context for the K90 sparsity metric (DR-003A).

    Scientific inputs:
        attributions: immutable tuple of AttributionRecord objects
        threshold:    L1 mass coverage threshold, frozen at 0.90 per DR-003A.
                      Must satisfy 0 < threshold <= 1 and be finite.

    Missing/negative/non-finite runtime_ms values in individual records are handled
    inside K90Sparsity.compute() as structured FailureRecord outputs — not rejected here.
    The attribution collection is stored as an immutable tuple for nested immutability.
    """

    attributions: tuple  # tuple[AttributionRecord, ...]
    threshold: float = 0.90

    @field_validator("attributions", mode="before")
    @classmethod
    def coerce_attributions(cls, value: object) -> tuple:  # noqa: ANN401
        """Accept any sequence; coerce to immutable tuple and validate element types."""
        from collections.abc import Sequence as _Seq

        from explaincheck.contracts import AttributionRecord

        if not isinstance(value, _Seq) or isinstance(value, (str, bytes)):
            raise ValueError("attributions must be a sequence of AttributionRecord")
        records: tuple[object, ...] = tuple(value)
        if not records:
            raise ValueError("SparsityContext: attributions must not be empty")
        for i, rec in enumerate(records):
            if not isinstance(rec, AttributionRecord):
                raise ValueError(
                    f"attributions[{i}] must be AttributionRecord, got {type(rec).__name__}"
                )
        return records

    @field_validator("threshold")
    @classmethod
    def threshold_valid(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"threshold must be finite, got {v}")
        if not (0.0 < v <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {v}")
        return v


class RuntimeContext(BaseMetricContext):
    """
    Context for the Runtime metric (DR-003A).

    Scientific inputs:
        attributions: immutable tuple of AttributionRecord objects.
                      Each record must carry a runtime_ms field recorded by the explainer adapter.

    Missing, negative or non-finite runtime_ms values are converted to structured FailureRecord
    objects inside RuntimeMetric.compute() — they are not rejected at context construction time.
    The attribution collection is stored as an immutable tuple for nested immutability.
    """

    attributions: tuple  # tuple[AttributionRecord, ...]

    @field_validator("attributions", mode="before")
    @classmethod
    def coerce_attributions(cls, value: object) -> tuple:  # noqa: ANN401
        """Accept any sequence; coerce to immutable tuple and validate element types."""
        from collections.abc import Sequence as _Seq

        from explaincheck.contracts import AttributionRecord

        if not isinstance(value, _Seq) or isinstance(value, (str, bytes)):
            raise ValueError("attributions must be a sequence of AttributionRecord")
        records: tuple[object, ...] = tuple(value)
        if not records:
            raise ValueError("RuntimeContext: attributions must not be empty")
        for i, rec in enumerate(records):
            if not isinstance(rec, AttributionRecord):
                raise ValueError(
                    f"attributions[{i}] must be AttributionRecord, got {type(rec).__name__}"
                )
        return records
