"""
ExplainCheck — abstract base class for explainer adapters.

Every explainer adapter must:
  1. Record background dataset hash, sampling budget, target class, seed, library version.
  2. Keep local and global explanation pipelines separate.
  3. Return AttributionRecord objects (not raw arrays) to enforce schema.
  4. Report runtime per sample.
  5. Catch and propagate failures as FailureRecord objects — never silently swallow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from explaincheck.contracts import (
    AttributionRecord,
    ExplainerName,
    ExplainerType,
    FailureRecord,
    ModelFamily,
)


class BaseExplainer(ABC):
    """Abstract interface for all explainer adapters."""

    name: ExplainerName
    explainer_type: ExplainerType = ExplainerType.LOCAL

    @abstractmethod
    def fit(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: list[str],
        *,
        seed: int,
    ) -> None:
        """
        Initialise the explainer with a fitted model and background data.
        X_background must come from the TRAINING split only.
        Must record the background dataset hash.
        """

    @abstractmethod
    def explain(
        self,
        X: np.ndarray,
        *,
        run_id: str,
        dataset: str,
        seed: int,
        model_family: ModelFamily,
        model_hash: str,
        sample_ids: list[str],
        protocol_version: str,
    ) -> list[AttributionRecord | FailureRecord]:
        """
        Explain all rows in X. Return one record per row.
        Failures must be returned as FailureRecord, not raised.
        """

    @abstractmethod
    def library_version(self) -> str:
        """Return the version string of the underlying library."""

    @abstractmethod
    def explainer_config(self) -> dict[str, Any]:
        """
        Return the complete configuration used by this explainer instance:
        background_n_samples, n_samples, target_class, kernel_width,
        discretize_continuous, seeds, etc.
        """
