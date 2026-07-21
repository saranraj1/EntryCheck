"""
ExplainCheck â€” Prediction-conditioned Top-k Jaccard stability metric.

Migrated from Phase 0 (run_phase0.py) without changing scientific definitions.

Scientific definition (frozen from Phase 0):
    For each sample, generate a Gaussian-perturbed copy (sigma=0.05).
    Discard pairs where the predicted class changes (prediction preservation).
    Stability = |top_k(A) âˆ© top_k(A')| / |top_k(A) âˆª top_k(A')|
    where A is the original attribution and A' is the attribution on perturbed input.

Prediction preservation is MANDATORY per protocol-v1.yaml.
Record the rejection rate (proportion of pairs where class changed).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

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
from explaincheck.metrics.contexts import StabilityContext
from explaincheck.provenance import utc_now_iso


def _topk_idx(row: np.ndarray, k: int) -> np.ndarray:
    """Indices of k largest-absolute-value features. Stable sort (matches Phase 0)."""
    return np.argsort(-np.abs(row), kind="stable")[:k]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Top-k Jaccard similarity. Migrated from Phase 0."""
    sa = set(_topk_idx(a, k).tolist())
    sb = set(_topk_idx(b, k).tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class TopKJaccardStability(BaseMetric[StabilityContext]):
    """
    Prediction-conditioned Top-k Jaccard stability.

    Direction: higher = more stable (same top-k features across perturbations).
    Range: [0, 1].
    """

    family = MetricFamily.STABILITY
    name = "stability_top_k_jaccard"
    direction = "higher_is_better"
    value_range = (0.0, 1.0)
    requires_prediction_preservation = True
    aggregation_method = "mean_over_preserved_pairs"

    def __init__(self, k: int = 3, sigma: float = 0.05) -> None:
        self.k = k
        self.sigma = sigma

    @property
    def assumptions(self) -> list[str]:
        return [
            "Prediction preservation is mandatory: pairs where class changes are discarded.",
            "Perturbation is additive Gaussian with sigma specified in parameters.",
            "Top-k is determined by absolute attribution value, stable sort.",
            "Phase 0 definition: Jaccard(top_k(A), top_k(A')) averaged over preserved pairs.",
        ]

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "sigma": self.sigma,
            "perturbation": "additive_gaussian",
            "prediction_preservation": True,
            "aggregation": "mean_over_preserved_pairs",
        }

    def compute(self, context: StabilityContext) -> list[MetricResult | FailureRecord]:
        """
        Compute Top-k Jaccard stability for each sample.

        Uses seed + 9000 for the perturbation RNG (identical to Phase 0).
        """
        self.validate_context(context)
        attributions: list[AttributionRecord] = context.attributions

        rng = np.random.default_rng(context.seed + 9000)
        n = len(attributions)
        Xp = context.X[:n] + rng.normal(0, context.sigma, size=context.X[:n].shape)

        # Predictions on original and perturbed inputs
        pred_orig = (_sigmoid(context.X[:n] @ context.weights + context.bias) >= 0.5).astype(int)
        pred_pert = (_sigmoid(Xp @ context.weights + context.bias) >= 0.5).astype(int)
        preserved_mask = pred_orig == pred_pert

        n_total = len(attributions)
        n_rejected = int((~preserved_mask).sum())
        prediction_preservation_rate = float(preserved_mask.mean())

        results: list[MetricResult | FailureRecord] = []

        for i, rec in enumerate(attributions):
            t0 = time.perf_counter()
            try:
                attr_orig = np.array(rec.attribution)

                # Compute perturbed attribution using the same explainer logic
                # For exact_linear: A' = w * (Xp - baseline)
                if rec.explainer == ExplainerName.EXACT_LINEAR:
                    attr_pert = (Xp[i] - context.baseline) * context.weights
                elif rec.explainer == ExplainerName.RANDOMIZED_NEGATIVE_CONTROL:
                    exact_pert = (Xp[i] - context.baseline) * context.weights
                    rng2 = np.random.default_rng(context.seed + 1)
                    attr_pert = exact_pert[rng2.permutation(exact_pert.size)]
                else:
                    raise ValueError(
                        f"Explainer {rec.explainer} not supported in this stability metric path. "
                        "Use the general stability runner for SHAP/LIME."
                    )

                if not preserved_mask[i]:
                    rt = (time.perf_counter() - t0) * 1000
                    results.append(
                        MetricResult(
                            run_id=context.run_id,
                            protocol_version=context.protocol_version,
                            dataset=context.dataset,
                            dataset_version=context.dataset_version,
                            split_hash=context.split_hash,
                            model_family=ModelFamily(context.model_family),
                            model_hash=context.model_hash,
                            explainer=rec.explainer,
                            explainer_version=rec.explainer_version,
                            seed=context.seed,
                            sample_id=rec.sample_id,
                            metric_family=self.family,
                            metric_name=self.name,
                            metric_k=self.k,
                            stressor=context.stressor,
                            stress_level=context.stress_level,
                            subgroup=context.subgroup,
                            subgroup_value=context.subgroup_value,
                            prediction_preservation_status=PredictionPreservationStatus.NOT_PRESERVED,
                            n_perturbations_total=n_total,
                            n_perturbations_rejected=n_rejected,
                            prediction_preservation_rate=prediction_preservation_rate,
                            estimate=float("nan"),
                            runtime_ms=rt,
                            status=RunStatus.EXCLUDED,
                            failure_reason="Prediction changed after perturbation (excluded by prediction-preservation filter).",
                        )
                    )
                    continue

                score = jaccard(attr_orig, attr_pert, self.k)
                rt = (time.perf_counter() - t0) * 1000
                results.append(
                    MetricResult(
                        run_id=context.run_id,
                        protocol_version=context.protocol_version,
                        dataset=context.dataset,
                        dataset_version=context.dataset_version,
                        split_hash=context.split_hash,
                        model_family=ModelFamily(context.model_family),
                        model_hash=context.model_hash,
                        explainer=rec.explainer,
                        explainer_version=rec.explainer_version,
                        seed=context.seed,
                        sample_id=rec.sample_id,
                        metric_family=self.family,
                        metric_name=self.name,
                        metric_k=self.k,
                        stressor=context.stressor,
                        stress_level=context.stress_level,
                        subgroup=context.subgroup,
                        subgroup_value=context.subgroup_value,
                        prediction_preservation_status=PredictionPreservationStatus.PRESERVED,
                        n_perturbations_total=n_total,
                        n_perturbations_rejected=n_rejected,
                        prediction_preservation_rate=prediction_preservation_rate,
                        estimate=score,
                        runtime_ms=rt,
                        status=RunStatus.SUCCESS,
                    )
                )

            except Exception as exc:
                rt = (time.perf_counter() - t0) * 1000
                results.append(
                    FailureRecord(
                        run_id=context.run_id,
                        timestamp=utc_now_iso(),
                        dataset=context.dataset,
                        model_family=ModelFamily(context.model_family),
                        explainer=rec.explainer,
                        metric_name=self.name,
                        seed=context.seed,
                        failure_reason=str(exc),
                        is_deterministic=True,
                        excluded=False,
                    )
                )

        return results
