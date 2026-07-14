"""
ExplainCheck — Scikit-learn Logistic Regression model adapter (Stage 3).

Reference adapter using deterministic solver. The custom Phase 0 gradient-descent
implementation remains available as the primary reproduction reference.

DR-003 §5: record solver, regularization, class weighting, iteration limit, convergence status.
Convergence warnings are structured, not console noise.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from explaincheck.contracts import ModelFamily, ModelRecord
from explaincheck.models.base import BaseModel


class SklearnLRAdapter(BaseModel):
    """
    Scikit-learn Logistic Regression adapter.

    Uses `lbfgs` solver (deterministic). Treats convergence warnings as
    structured metadata, not console noise.
    """

    family = ModelFamily.LOGISTIC_REGRESSION

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.class_weight = class_weight
        self.random_state = random_state
        self._model: LogisticRegression | None = None
        self._converged: bool | None = None
        self._n_iter: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> ModelRecord:
        t0 = time.perf_counter()
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            class_weight=self.class_weight,
            random_state=self.random_state if self.random_state is not None else seed,
            n_jobs=1,
        )
        convergence_warning: str | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X, y)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    convergence_warning = str(w.message)

        self._model = clf
        self._converged = convergence_warning is None
        self._n_iter = int(clf.n_iter_[0])
        fit_ms = (time.perf_counter() - t0) * 1000

        h = hashlib.sha256()
        h.update(clf.coef_.tobytes())
        h.update(clf.intercept_.tobytes())
        model_hash = h.hexdigest()

        train_h = hashlib.sha256()
        train_h.update(X.tobytes())
        train_h.update(y.tobytes())

        import sklearn
        return ModelRecord(
            family=self.family,
            implementation="sklearn.linear_model.LogisticRegression",
            version=sklearn.__version__,
            hyperparameters=self.get_hyperparameters(),
            dataset="(set by caller)",
            seed=seed,
            fit_ms=fit_ms,
            model_hash=model_hash,
            train_sha256=train_h.hexdigest(),
            notes=f"converged={self._converged}, n_iter={self._n_iter}"
            + (f", convergence_warning={convergence_warning!r}" if convergence_warning else ""),
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
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def library_version(self) -> str:
        import sklearn
        return sklearn.__version__

    @property
    def coef_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.coef_[0]

    @property
    def intercept_(self) -> float:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return float(self._model.intercept_[0])
