"""
ExplainCheck — abstract base class for dataset adapters.

Every dataset adapter must:
  1. Inherit from BaseDataset.
  2. Implement all abstract methods.
  3. Fit all preprocessing on training data only.
  4. Preserve raw column names and deterministic feature lineage after encoding.
  5. Expose the DatasetRecord provenance object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explaincheck.contracts import DatasetRecord, SplitRecord, TaskType


class BaseDataset(ABC):
    """Abstract interface for all dataset adapters."""

    # Subclasses must set these class-level attributes.
    name: str = ""
    doi: str = ""
    url: str = ""
    license: str = "CC BY 4.0"
    task: TaskType = TaskType.BINARY_CLASSIFICATION

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._raw: pd.DataFrame | None = None
        self._X_train: np.ndarray | None = None
        self._X_test: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._record: DatasetRecord | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load raw data from data_dir. Must populate self._raw."""

    @abstractmethod
    def split(self, seed: int) -> SplitRecord:
        """
        Split data deterministically by seed.
        Must fit NO transformations — raw split only.
        Returns a SplitRecord with hashes of both halves.
        """

    @abstractmethod
    def preprocess(self) -> None:
        """
        Fit all preprocessing on training data and transform both halves.
        Must NOT touch test data until after training fit.
        Must record feature lineage.
        """

    @abstractmethod
    def provenance(self) -> DatasetRecord:
        """Return the DatasetRecord provenance object."""

    # ------------------------------------------------------------------
    # Shared accessors (populated after split + preprocess)
    # ------------------------------------------------------------------

    @property
    def X_train(self) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Call split() and preprocess() first.")
        return self._X_train

    @property
    def X_test(self) -> np.ndarray:
        if self._X_test is None:
            raise RuntimeError("Call split() and preprocess() first.")
        return self._X_test

    @property
    def y_train(self) -> np.ndarray:
        if self._y_train is None:
            raise RuntimeError("Call split() and preprocess() first.")
        return self._y_train

    @property
    def y_test(self) -> np.ndarray:
        if self._y_test is None:
            raise RuntimeError("Call split() and preprocess() first.")
        return self._y_test

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Call preprocess() first.")
        return self._feature_names

    # ------------------------------------------------------------------
    # Subgroup helpers (optional — override in subclasses)
    # ------------------------------------------------------------------

    def subgroup_masks(self) -> dict[str, np.ndarray]:
        """
        Return Boolean masks for each subgroup over the test split.
        Default: no subgroups. Override in dataset-specific adapters.
        """
        return {}
