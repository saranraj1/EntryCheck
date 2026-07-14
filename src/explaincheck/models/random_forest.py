"""
ExplainCheck — Random Forest classifier adapter (Stage 3).

DR-003 §5: pass random_state, n_jobs=1, record n_estimators/depth/criterion/class_weight.
Test probability and class-label consistency.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from explaincheck.contracts import ModelFamily, ModelRecord
from explaincheck.models.base import BaseModel


class RandomForestAdapter(BaseModel):
    """
    Scikit-learn RandomForestClassifier adapter.

    n_jobs=1 for reproducible runtime measurements.
    """

    family = ModelFamily.RANDOM_FOREST

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        criterion: str = "gini",
        class_weight: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state
        self._model: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> ModelRecord:
        t0 = time.perf_counter()
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            criterion=self.criterion,
            class_weight=self.class_weight,
            random_state=self.random_state if self.random_state is not None else seed,
            n_jobs=1,
        )
        clf.fit(X, y)
        self._model = clf
        fit_ms = (time.perf_counter() - t0) * 1000

        # Model hash: hash the leaf values of all estimators as a fingerprint
        h = hashlib.sha256()
        for est in clf.estimators_:
            h.update(est.tree_.value.tobytes())
        model_hash = h.hexdigest()

        train_h = hashlib.sha256()
        train_h.update(X.tobytes())
        train_h.update(y.tobytes())

        import sklearn
        return ModelRecord(
            family=self.family,
            implementation="sklearn.ensemble.RandomForestClassifier",
            version=sklearn.__version__,
            hyperparameters=self.get_hyperparameters(),
            dataset="(set by caller)",
            seed=seed,
            fit_ms=fit_ms,
            model_hash=model_hash,
            train_sha256=train_h.hexdigest(),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(X)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": 1,
        }

    def library_version(self) -> str:
        import sklearn
        return sklearn.__version__

    @property
    def sklearn_model(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model
