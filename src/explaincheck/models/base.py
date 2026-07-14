"""
ExplainCheck — abstract base class for model adapters.

Rules (from DR-001-PHASE1):
- n_jobs=1 during controlled runtime measurements.
- Pass every random seed explicitly.
- Record all hyperparameters and library version in the ModelRecord.
- XGBoost: CPU only; no GPU; version frozen at 2.1.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from explaincheck.contracts import ModelFamily, ModelRecord


class BaseModel(ABC):
    """Abstract interface for all model adapters."""

    family: ModelFamily

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> ModelRecord:
        """
        Fit the model. Return a ModelRecord with provenance.
        Must not use any information from the test split.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-probability array of shape (n_samples, n_classes)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels of shape (n_samples,)."""

    @abstractmethod
    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the complete hyperparameter dict used for this model."""

    @abstractmethod
    def library_version(self) -> str:
        """Return the library version string (e.g. '2.1.4' for XGBoost)."""
