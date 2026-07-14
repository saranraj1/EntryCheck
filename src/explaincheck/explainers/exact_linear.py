"""
ExplainCheck — Exact-linear attribution and randomized negative control.

Migrated from Phase 0 (run_phase0.py) without changing scientific definitions.

Scientific definitions (frozen from Phase 0):
    Exact-linear:   attribution_i = w_i * (x_i - baseline_i)
                    where baseline = training-set mean
    Negative control: within-instance random permutation of exact attributions.
                      Not a competitive explainer.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from explaincheck.contracts import (
    AttributionRecord,
    DataSplit,
    ExplainerName,
    ExplainerType,
    FailureRecord,
    ModelFamily,
    RunStatus,
)
from explaincheck.explainers.base import BaseExplainer
from explaincheck.models.logistic_regression import LogisticRegressionAdapter
from explaincheck.provenance import utc_now_iso


class ExactLinearExplainer(BaseExplainer):
    """
    Exact-linear attribution: w_i * (x_i - baseline_i).

    This is a control reference — not a competitive explanation method.
    Only valid for logistic regression where weights are interpretable.
    """

    name = ExplainerName.EXACT_LINEAR
    explainer_type = ExplainerType.CONTROL_REFERENCE

    def __init__(self) -> None:
        self._weights: np.ndarray | None = None
        self._baseline: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._background_hash: str | None = None

    def fit(
        self,
        model: LogisticRegressionAdapter,
        X_background: np.ndarray,
        feature_names: list[str],
        *,
        seed: int,
    ) -> None:
        """
        Initialise with fitted LR weights and training-set mean as baseline.
        X_background must be the TRAINING split only.
        """
        from explaincheck.provenance import hash_array

        self._weights = model.weights.copy()
        self._baseline = X_background.mean(axis=0)
        self._feature_names = list(feature_names)
        self._background_hash = hash_array(X_background)

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
        model: LogisticRegressionAdapter | None = None,
    ) -> list[AttributionRecord | FailureRecord]:
        if self._weights is None or self._baseline is None:
            raise RuntimeError("Call fit() first.")

        records: list[AttributionRecord | FailureRecord] = []
        for i, sid in enumerate(sample_ids):
            t0 = time.perf_counter()
            try:
                attr = (X[i] - self._baseline) * self._weights
                if model is not None:
                    prob = float(model.predict_proba(X[i : i + 1])[0, 1])
                    pred_class = int(prob >= 0.5)
                else:
                    prob = float("nan")
                    pred_class = -1
                rt = (time.perf_counter() - t0) * 1000
                records.append(
                    AttributionRecord(
                        schema_version="1.0.0",
                        run_id=run_id,
                        protocol_version=protocol_version,
                        sample_id=sid,
                        dataset=dataset,
                        seed=seed,
                        split=DataSplit.TEST,
                        model_family=model_family,
                        model_hash=model_hash,
                        explainer=self.name,
                        explainer_type=self.explainer_type,
                        explainer_version="1.0.0-phase0",
                        feature_names=self._feature_names or [],
                        attribution=attr.tolist(),
                        prediction_class=pred_class,
                        prediction_probability=prob,
                        runtime_ms=rt,
                        success=True,
                    )
                )
            except Exception as exc:
                rt = (time.perf_counter() - t0) * 1000
                records.append(
                    FailureRecord(
                        run_id=run_id,
                        timestamp=utc_now_iso(),
                        dataset=dataset,
                        model_family=model_family,
                        explainer=self.name,
                        seed=seed,
                        failure_reason=str(exc),
                        is_deterministic=True,
                        excluded=False,
                    )
                )
        return records

    def library_version(self) -> str:
        return "explaincheck-exact-linear-1.0.0"

    def explainer_config(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": self.explainer_type.value,
            "baseline": "training_set_mean",
            "background_hash": self._background_hash,
            "version": self.library_version(),
            "notes": "Analytic reference; not a competitive explainer.",
        }


class RandomizedNegativeControl(BaseExplainer):
    """
    Within-instance random permutation of exact-linear attributions.

    This is a negative control — not a competitive explanation method.
    Expected to score lower than any valid explainer on fidelity and stability.
    """

    name = ExplainerName.RANDOMIZED_NEGATIVE_CONTROL
    explainer_type = ExplainerType.CONTROL_NEGATIVE

    def __init__(self) -> None:
        self._exact: ExactLinearExplainer | None = None
        self._background_hash: str | None = None

    def fit(
        self,
        model: LogisticRegressionAdapter,
        X_background: np.ndarray,
        feature_names: list[str],
        *,
        seed: int,
    ) -> None:
        """Initialise the underlying exact explainer first."""
        self._exact = ExactLinearExplainer()
        self._exact.fit(model, X_background, feature_names, seed=seed)
        self._background_hash = self._exact._background_hash

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
        model: LogisticRegressionAdapter | None = None,
    ) -> list[AttributionRecord | FailureRecord]:
        if self._exact is None:
            raise RuntimeError("Call fit() first.")

        # Get exact attributions first
        exact_records = self._exact.explain(
            X,
            run_id=run_id,
            dataset=dataset,
            seed=seed,
            model_family=model_family,
            model_hash=model_hash,
            sample_ids=sample_ids,
            protocol_version=protocol_version,
            model=model,
        )

        rng = np.random.default_rng(seed)
        records: list[AttributionRecord | FailureRecord] = []
        for rec in exact_records:
            if isinstance(rec, FailureRecord):
                records.append(rec)
                continue
            t0 = time.perf_counter()
            arr = np.array(rec.attribution)
            shuffled = arr[rng.permutation(arr.size)]
            rt = (time.perf_counter() - t0) * 1000
            records.append(
                AttributionRecord(
                    schema_version="1.0.0",
                    run_id=rec.run_id,
                    protocol_version=rec.protocol_version,
                    sample_id=rec.sample_id,
                    dataset=rec.dataset,
                    seed=rec.seed,
                    split=rec.split,
                    model_family=rec.model_family,
                    model_hash=rec.model_hash,
                    explainer=self.name,
                    explainer_type=self.explainer_type,
                    explainer_version="1.0.0-phase0",
                    feature_names=rec.feature_names,
                    attribution=shuffled.tolist(),
                    prediction_class=rec.prediction_class,
                    prediction_probability=rec.prediction_probability,
                    runtime_ms=rec.runtime_ms + rt,
                    success=True,
                )
            )
        return records

    def library_version(self) -> str:
        return "explaincheck-randomized-negative-control-1.0.0"

    def explainer_config(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "type": self.explainer_type.value,
            "method": "within_instance_permutation_of_exact_linear",
            "background_hash": self._background_hash,
            "version": self.library_version(),
            "notes": "Negative control; not a competitive explainer.",
        }
