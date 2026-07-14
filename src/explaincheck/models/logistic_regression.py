"""
ExplainCheck — Logistic Regression model adapter.

Migrated from Phase 0 (run_phase0.py) without changing scientific definitions.

Scientific definition (frozen from Phase 0):
    L2-regularized logistic regression fitted by full-batch gradient descent.
    Parameters: steps=1800, lr=0.12, l2=1e-3 (matches pilot.yaml).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from explaincheck.contracts import ModelFamily, ModelRecord
from explaincheck.models.base import BaseModel
from explaincheck.provenance import utc_now_iso


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. Identical to Phase 0."""
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionAdapter(BaseModel):
    """
    Full-batch gradient-descent L2-regularized logistic regression.

    Migrated directly from Phase 0. Parameters must match pilot.yaml to
    ensure Phase 0 reproduction within declared tolerance.
    """

    family = ModelFamily.LOGISTIC_REGRESSION

    def __init__(
        self,
        steps: int = 1800,
        lr: float = 0.12,
        l2: float = 1e-3,
    ) -> None:
        self.steps = steps
        self.lr = lr
        self.l2 = l2
        self._w: np.ndarray | None = None
        self._b: float | None = None
        self._fit_ms: float | None = None
        self._train_sha256: str | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> ModelRecord:
        """
        Fit by full-batch gradient descent.

        The seed parameter is accepted for interface compatibility but this
        implementation is deterministic given X, y — no stochastic component.
        """
        t0 = time.perf_counter()
        w = np.zeros(X.shape[1])
        b = 0.0
        n = len(y)
        y_f = y.astype(np.float64)

        for _ in range(self.steps):
            p = _sigmoid(X @ w + b)
            err = p - y_f
            grad_w = (X.T @ err) / n + self.l2 * w
            grad_b = float(np.mean(err))
            w -= self.lr * grad_w
            b -= self.lr * grad_b

        self._w = w
        self._b = b
        self._fit_ms = (time.perf_counter() - t0) * 1000

        # Hash the training data for provenance
        h = hashlib.sha256()
        h.update(X.tobytes())
        h.update(y.tobytes())
        self._train_sha256 = h.hexdigest()

        # Hash model parameters
        model_hash = hashlib.sha256(
            json.dumps({"w": w.tolist(), "b": b}, sort_keys=True).encode()
        ).hexdigest()

        import sklearn
        return ModelRecord(
            family=self.family,
            implementation="explaincheck.models.logistic_regression.LogisticRegressionAdapter",
            version=f"explaincheck-lr-phase0/{sklearn.__version__}",
            hyperparameters=self.get_hyperparameters(),
            dataset="(set by caller)",
            seed=seed,
            fit_ms=self._fit_ms,
            model_hash=model_hash,
            train_sha256=self._train_sha256,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._w is None or self._b is None:
            raise RuntimeError("Call fit() first.")
        prob_pos = _sigmoid(X @ self._w + self._b)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def get_hyperparameters(self) -> dict[str, Any]:
        return {"steps": self.steps, "lr": self.lr, "l2": self.l2}

    def library_version(self) -> str:
        return "explaincheck-custom-lr-phase0-1.0.0"

    # ------------------------------------------------------------------
    # Phase 0 helpers
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("Call fit() first.")
        return self._w

    @property
    def bias(self) -> float:
        if self._b is None:
            raise RuntimeError("Call fit() first.")
        return self._b

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
