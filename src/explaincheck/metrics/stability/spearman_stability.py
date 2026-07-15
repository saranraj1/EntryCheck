"""
ExplainCheck — Spearman rank correlation stability metric (Stage 4, DR-003A).

Scientific definition:
    For each sample, generate a Gaussian-perturbed copy (sigma=0.05).
    Discard pairs where the predicted class changes (prediction preservation).
    Spearman_rho(A, A') = Pearson correlation of rank-transformed A and A'.

    When both attribution vectors are constant (zero variance), rho = 1.0.
    When one is constant, other is not → rho = NaN → recorded as excluded.

Direction: higher = more stable.
Range: [-1, 1].
Prediction preservation: mandatory per protocol-v1.yaml.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from explaincheck.contracts import (
    AttributionRecord,
    ExplainerName,
    FailureRecord,
    MetricFamily,
    MetricResult,
    ModelFamily,
    PredictionPreservationStatus,
    RunStatus,
)
from explaincheck.metrics.base import BaseMetric
from explaincheck.metrics.stability.top_k_jaccard import _sigmoid
from explaincheck.provenance import utc_now_iso


def spearman_pair(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two dense attribution vectors.

    Edge cases:
    - Both constant → 1.0 (identical trivial attributions)
    - One constant, other not → NaN (undefined; caller should handle)

    Parameters
    ----------
    a, b : np.ndarray
        Dense attribution vectors of equal length.

    Raises
    ------
    ValueError
        If arrays have different shapes, are empty, or contain NaN/Inf.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Attribution contains NaN or Inf.")
    if len(a) == 0:
        raise ValueError("Empty attribution vectors.")
    if np.std(a) == 0.0 and np.std(b) == 0.0:
        return 1.0
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


class SpearmanStability(BaseMetric):
    """
    Prediction-conditioned Spearman rank correlation stability.

    Direction: higher = more stable (attribution rank orders preserved across perturbations).
    """

    family = MetricFamily.STABILITY
    name = "spearman_stability"
    direction = "higher_is_better"
    value_range = (-1.0, 1.0)
    requires_prediction_preservation = True
    aggregation_method = "mean"

    def __init__(
        self,
        *,
        sigma: float = 0.05,
        n_perturbations: int = 5,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if n_perturbations < 1:
            raise ValueError(f"n_perturbations must be >= 1, got {n_perturbations}")
        self._sigma = sigma
        self._n_perturbations = n_perturbations

    @property
    def assumptions(self) -> list[str]:
        return [
            "Attribution vectors are dense and finite.",
            "Prediction preservation is enforced: pairs where predicted class changes are excluded.",
            "Perturbation is Gaussian with sigma=0.05 (frozen from Phase 0 protocol).",
            "Spearman rho is undefined when one vector has zero variance — recorded as excluded.",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "sigma": self._sigma,
            "n_perturbations": self._n_perturbations,
        }

    def compute(  # type: ignore[override]
        self,
        attributions: list[AttributionRecord],
        *,
        run_id: str,
        protocol_version: str,
        dataset: str,
        dataset_version: str,
        split_hash: str,
        model_family: ModelFamily,
        model_hash: str,
        seed: int,
        stressor: str | None = None,
        stress_level: str | None = None,
        subgroup: str | None = None,
        subgroup_value: str | None = None,
        weights: np.ndarray,
        bias: float,
        baseline: np.ndarray,
        X: np.ndarray,
        **kwargs: Any,
    ) -> list[MetricResult | FailureRecord]:
        """Compute Spearman stability using analytic attribution pairs (exact_linear)."""
        self.validate_attributions(attributions)

        rng = np.random.default_rng(seed + 9000)
        n = len(attributions)
        Xp = X[:n] + rng.normal(0, self._sigma, size=X[:n].shape)

        pred_orig = (_sigmoid(X[:n] @ weights + bias) >= 0.5).astype(int)
        pred_pert = (_sigmoid(Xp @ weights + bias) >= 0.5).astype(int)
        preserved_mask = pred_orig == pred_pert

        n_total = n
        n_rejected = int((~preserved_mask).sum())

        results: list[MetricResult | FailureRecord] = []

        for i, rec in enumerate(attributions):
            t0 = time.perf_counter()
            try:
                attr_orig = np.array(rec.attribution, dtype=float)

                if rec.explainer == ExplainerName.EXACT_LINEAR:
                    attr_pert = (Xp[i] - baseline) * weights
                elif rec.explainer == ExplainerName.RANDOMIZED_NEGATIVE_CONTROL:
                    exact_pert = (Xp[i] - baseline) * weights
                    rng2 = np.random.default_rng(seed + 1)
                    attr_pert = exact_pert[rng2.permutation(exact_pert.size)]
                else:
                    raise ValueError(
                        f"Explainer {rec.explainer} not supported in this stability path. "
                        "Use the general stability runner for SHAP/LIME."
                    )

                if not preserved_mask[i]:
                    rt = (time.perf_counter() - t0) * 1000
                    results.append(
                        MetricResult(
                            run_id=run_id,
                            protocol_version=protocol_version,
                            dataset=dataset,
                            dataset_version=dataset_version,
                            split_hash=split_hash,
                            model_family=model_family,
                            model_hash=model_hash,
                            explainer=rec.explainer,
                            explainer_version=rec.explainer_version,
                            seed=seed,
                            sample_id=rec.sample_id,
                            metric_family=self.family,
                            metric_name=self.name,
                            stressor=stressor,
                            stress_level=stress_level,
                            subgroup=subgroup,
                            subgroup_value=subgroup_value,
                            prediction_preservation_status=PredictionPreservationStatus.NOT_PRESERVED,
                            n_perturbations_total=n_total,
                            n_perturbations_rejected=n_rejected,
                            estimate=float("nan"),
                            runtime_ms=rt,
                            status=RunStatus.EXCLUDED,
                            failure_reason="Prediction changed after perturbation (excluded by prediction-preservation filter).",
                        )
                    )
                    continue

                rho = spearman_pair(attr_orig, attr_pert)
                rt = (time.perf_counter() - t0) * 1000

                if not np.isfinite(rho):
                    results.append(
                        MetricResult(
                            run_id=run_id,
                            protocol_version=protocol_version,
                            dataset=dataset,
                            dataset_version=dataset_version,
                            split_hash=split_hash,
                            model_family=model_family,
                            model_hash=model_hash,
                            explainer=rec.explainer,
                            explainer_version=rec.explainer_version,
                            seed=seed,
                            sample_id=rec.sample_id,
                            metric_family=self.family,
                            metric_name=self.name,
                            stressor=stressor,
                            stress_level=stress_level,
                            subgroup=subgroup,
                            subgroup_value=subgroup_value,
                            prediction_preservation_status=PredictionPreservationStatus.PRESERVED,
                            n_perturbations_total=n_total,
                            n_perturbations_rejected=n_rejected,
                            estimate=float("nan"),
                            runtime_ms=rt,
                            status=RunStatus.EXCLUDED,
                            failure_reason="Spearman rho undefined: one attribution vector has zero variance.",
                        )
                    )
                else:
                    results.append(
                        MetricResult(
                            run_id=run_id,
                            protocol_version=protocol_version,
                            dataset=dataset,
                            dataset_version=dataset_version,
                            split_hash=split_hash,
                            model_family=model_family,
                            model_hash=model_hash,
                            explainer=rec.explainer,
                            explainer_version=rec.explainer_version,
                            seed=seed,
                            sample_id=rec.sample_id,
                            metric_family=self.family,
                            metric_name=self.name,
                            stressor=stressor,
                            stress_level=stress_level,
                            subgroup=subgroup,
                            subgroup_value=subgroup_value,
                            prediction_preservation_status=PredictionPreservationStatus.PRESERVED,
                            n_perturbations_total=n_total,
                            n_perturbations_rejected=n_rejected,
                            estimate=rho,
                            runtime_ms=rt,
                            status=RunStatus.SUCCESS,
                        )
                    )

            except Exception as exc:
                rt = (time.perf_counter() - t0) * 1000
                results.append(
                    FailureRecord(
                        run_id=run_id,
                        timestamp=utc_now_iso(),
                        dataset=dataset,
                        model_family=model_family,
                        explainer=rec.explainer,
                        metric_name=self.name,
                        seed=seed,
                        failure_reason=str(exc),
                        is_deterministic=True,
                        excluded=False,
                    )
                )

        return results
