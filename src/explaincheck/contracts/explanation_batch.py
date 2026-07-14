"""
ExplainCheck — Typed ExplanationBatch contract.

Every Stage 3 explainer adapter must return this object.
Rejected if any rejection condition in DR-003 §4 is met.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from explaincheck.contracts import ExplainerName, ExplainerType, ModelFamily


class ExplanationBatch(BaseModel):
    """
    Common typed explanation output contract (DR-003 §4).

    Every explainer adapter must produce one ExplanationBatch per call.
    Rejection conditions are enforced by validators.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Provenance
    run_id: str
    protocol_version: str
    dataset: str
    seed: int

    # Model
    model_family: ModelFamily
    model_hash: str

    # Explainer
    explainer: ExplainerName
    explainer_type: ExplainerType
    explainer_version: str
    explainer_params: dict[str, Any]

    # Output space
    output_space: str  # e.g. "probability", "log_odds", "raw_margin"
    target_class: int

    # Data
    sample_ids: list[str]
    feature_names: list[str]

    # Core outputs
    attributions: list[list[float]]  # shape: (n_samples, n_features)
    base_values: list[float] | None  # shape: (n_samples,), None if not available
    predictions: list[float]  # shape: (n_samples,), probability of target_class
    prediction_classes: list[int]  # shape: (n_samples,)

    # Background / background provenance
    background_hash: str | None  # SHA-256 of training background rows used
    background_n_rows: int | None  # number of background rows

    # Runtime
    runtime_ms: float

    # Status
    success: bool
    warnings: list[str]
    failure_reason: str | None

    # ---------------------------------------------------------------------------
    # Rejection validators (DR-003 §4)
    # ---------------------------------------------------------------------------

    @field_validator("attributions")
    @classmethod
    def no_nan_or_inf(cls, v: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)) or val != val or val in (float("inf"), float("-inf")):
                    raise ValueError(
                        f"Attribution contains NaN or Inf at sample {i}, feature {j}. "
                        "Batch rejected per DR-003 §4."
                    )
        return v

    @field_validator("output_space")
    @classmethod
    def output_space_specified(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("output_space must be specified. Batch rejected per DR-003 §4.")
        return v

    @field_validator("model_hash")
    @classmethod
    def model_hash_present(cls, v: str) -> str:
        if not v or len(v) < 8:
            raise ValueError("model_hash must be present. Batch rejected per DR-003 §4.")
        return v

    @model_validator(mode="after")
    def validate_attribution_width(self) -> ExplanationBatch:
        n_features = len(self.feature_names)
        for i, row in enumerate(self.attributions):
            if len(row) != n_features:
                raise ValueError(
                    f"Attribution width {len(row)} != feature_names width {n_features} "
                    f"at sample {i}. Batch rejected per DR-003 §4 (feature-schema mismatch)."
                )
        return self

    @model_validator(mode="after")
    def validate_sample_consistency(self) -> ExplanationBatch:
        n = len(self.sample_ids)
        if len(self.attributions) != n:
            raise ValueError("len(attributions) != len(sample_ids)")
        if len(self.predictions) != n:
            raise ValueError("len(predictions) != len(sample_ids)")
        if len(self.prediction_classes) != n:
            raise ValueError("len(prediction_classes) != len(sample_ids)")
        if self.base_values is not None and len(self.base_values) != n:
            raise ValueError("len(base_values) != len(sample_ids)")
        return self

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def attribution_matrix(self) -> np.ndarray:
        """Return (n_samples, n_features) float64 numpy array."""
        return np.array(self.attributions, dtype=np.float64)

    def additivity_residuals(self, logits: np.ndarray | None = None) -> np.ndarray | None:
        """
        For SHAP: residual = (base_value + sum(attributions)) - logit.
        Returns None if base_values is None (not applicable for LIME etc.).
        Returns (n_samples,) array if base_values are present.
        """
        if self.base_values is None:
            return None
        if logits is None:
            return None
        A = self.attribution_matrix()
        totals = np.array(self.base_values) + A.sum(axis=1)
        return totals - logits

    @staticmethod
    def hash_background(X_background: np.ndarray) -> str:
        """Compute reproducible SHA-256 hash of a background data array."""
        return hashlib.sha256(X_background.tobytes()).hexdigest()
