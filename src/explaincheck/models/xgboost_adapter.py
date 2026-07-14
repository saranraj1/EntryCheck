"""
ExplainCheck — XGBoost classifier adapter (Stage 3).

DR-003 §5: frozen xgboost==2.1.4, CPU execution, n_jobs=1 (via nthread),
record objective/eval_metric/tree_method/hyperparameters. No early stopping
in basic deterministic adapter tests.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import xgboost as xgb

from explaincheck.contracts import ModelFamily, ModelRecord
from explaincheck.models.base import BaseModel


class XGBoostAdapter(BaseModel):
    """
    XGBoost 2.1.4 (frozen) classifier adapter.

    CPU-only. n_jobs=1 via nthread for controlled timing.
    No early stopping in basic mode.
    """

    family = ModelFamily.XGBOOST

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.device = device
        self.default_seed = seed
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> ModelRecord:
        t0 = time.perf_counter()
        clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            device=self.device,
            random_state=seed,
            nthread=1,
            verbosity=0,
        )
        clf.fit(X, y)
        self._model = clf
        fit_ms = (time.perf_counter() - t0) * 1000

        # Model hash: hash the JSON booster dump
        booster_json = clf.get_booster().save_raw(raw_format="json")
        model_hash = hashlib.sha256(booster_json).hexdigest()

        train_h = hashlib.sha256()
        train_h.update(X.tobytes())
        train_h.update(y.tobytes())

        return ModelRecord(
            family=self.family,
            implementation="xgboost.XGBClassifier",
            version=xgb.__version__,
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
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "device": self.device,
            "nthread": 1,
        }

    def library_version(self) -> str:
        return xgb.__version__

    @property
    def sklearn_model(self) -> xgb.XGBClassifier:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model
