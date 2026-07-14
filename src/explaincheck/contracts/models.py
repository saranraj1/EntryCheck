"""
ExplainCheck typed contracts — Pydantic data models.

These models define the schema for every data object exchanged in the pipeline.
Schema version is embedded in every model that crosses a module boundary.

Rules:
- All models are immutable (model_config = frozen).
- Fields have explicit types; no bare Any except where unavoidable.
- Add new fields at the end to preserve backward schema compatibility.
- Breaking changes require a schema version bump.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from explaincheck.contracts.enums import (
    DataSplit,
    ExplainerName,
    ExplainerType,
    MetricFamily,
    ModelFamily,
    PredictionPreservationStatus,
    RunLabel,
    RunStatus,
    TaskType,
)

_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetRecord(BaseModel):
    """Provenance record for one dataset snapshot."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    name: str
    version: str
    doi: str
    url: str
    license: str
    retrieval_date: str  # ISO-8601 date string
    sha256: str
    n_rows: int
    n_features: int
    target_column: str
    task: TaskType
    split: DataSplit | None = None
    split_hash: str | None = None
    notes: str = ""


class SplitRecord(BaseModel):
    """Hashes for one train/test split."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    dataset: str
    seed: int
    train_sha256: str
    test_sha256: str
    train_n: int
    test_n: int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelRecord(BaseModel):
    """Provenance record for one fitted model."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    family: ModelFamily
    implementation: str      # fully-qualified class name
    version: str             # library version string
    hyperparameters: dict[str, Any]
    dataset: str
    seed: int
    fit_ms: float
    model_hash: str          # hash of serialised model parameters
    train_sha256: str
    notes: str | None = None  # structured metadata (e.g. convergence status)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class AttributionRecord(BaseModel):
    """One local attribution vector for one sample."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    protocol_version: str
    sample_id: str
    dataset: str
    seed: int
    split: DataSplit
    model_family: ModelFamily
    model_hash: str
    explainer: ExplainerName
    explainer_type: ExplainerType
    explainer_version: str
    feature_names: list[str]
    attribution: list[float]
    prediction_class: int
    prediction_probability: float
    runtime_ms: float
    success: bool
    failure_reason: str | None = None

    @field_validator("attribution")
    @classmethod
    def attributions_finite(cls, v: list[float]) -> list[float]:
        import math
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Attribution vector contains NaN or Inf.")
        return v


# ---------------------------------------------------------------------------
# Metric result
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """One metric estimate for one (dataset, model, explainer, seed, stressor) cell."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    protocol_version: str

    # Provenance dimensions
    dataset: str
    dataset_version: str
    split_hash: str
    model_family: ModelFamily
    model_hash: str
    explainer: ExplainerName
    explainer_version: str
    seed: int
    sample_id: str | None = None

    # Metric identity
    metric_family: MetricFamily
    metric_name: str           # e.g. "deletion_fidelity_aopc"
    metric_variant: str | None = None   # e.g. "top_k_jaccard" for stability
    metric_k: int | None = None        # k parameter for top-k metrics

    # Stressor dimensions
    stressor: str | None = None
    stress_level: str | None = None    # e.g. "0.3" for correlation rho

    # Subgroup dimension
    subgroup: str | None = None
    subgroup_value: str | None = None

    # Prediction preservation
    prediction_preservation_status: PredictionPreservationStatus = PredictionPreservationStatus.NOT_APPLICABLE
    n_perturbations_total: int | None = None
    n_perturbations_rejected: int | None = None

    # Result
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    n: int | None = None              # number of samples contributing

    # Runtime
    runtime_ms: float

    # Status
    status: RunStatus
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------

class FailureRecord(BaseModel):
    """One pipeline failure, preserved for the failures.csv output."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    timestamp: str          # ISO-8601 datetime string
    dataset: str
    model_family: ModelFamily | None = None
    explainer: ExplainerName | None = None
    metric_name: str | None = None
    seed: int
    failure_reason: str
    traceback: str | None = None
    is_deterministic: bool
    excluded: bool
    exclusion_justification: str | None = None  # must match frozen exclusion reasons


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

class ArtifactEntry(BaseModel):
    """SHA-256 hash and byte size for one output artifact."""

    model_config = ConfigDict(frozen=True)

    sha256: str
    bytes: int


class EnvironmentRecord(BaseModel):
    """Complete software environment snapshot."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    numpy_version: str
    pandas_version: str
    sklearn_version: str
    xgboost_version: str
    shap_version: str
    lime_version: str
    matplotlib_version: str
    pydantic_version: str
    platform: str
    cpu_count: int | None = None


class RunManifest(BaseModel):
    """Top-level provenance manifest for one frozen run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    study_id: str
    protocol_version: str
    run_label: RunLabel
    status: str                    # "pilot-not-confirmatory", "confirmatory", etc.
    config_hash: str               # SHA-256 of the YAML config used
    code_hash: str                 # git commit SHA or source hash
    created_at: str                # ISO-8601 datetime
    environment: EnvironmentRecord
    seeds: list[int]
    datasets: list[str]
    models: list[str]
    explainers: list[str]
    files: dict[str, ArtifactEntry]
    n_successes: int
    n_failures: int
    n_excluded: int
    elapsed_seconds: float
    limitations: list[str]
    osf_registration_url: str | None = None   # None until external registration
