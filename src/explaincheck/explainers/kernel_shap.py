"""
ExplainCheck — KernelSHAP explainer adapter (Stage 3).

DR-003 §3 requirements:
- Background data from TRAINING partition only
- Deterministic seeded sample of 50 training rows for Stage 3 tests
- Small fixed nsamples budget in CI
- Never use test rows as background
- Record background-row hashes and sampling parameters
- Model-agnostic: compatible with all model families
- Return common ExplanationBatch contract
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import shap

from explaincheck.contracts import ExplainerName, ExplainerType, ModelFamily
from explaincheck.contracts.explanation_batch import ExplanationBatch

# Stage 3 CI default budget (DR-003: "small fixed nsamples budget in CI")
STAGE3_CI_NSAMPLES = 64
STAGE3_CI_BACKGROUND_N = 50


def _sample_background(
    X_train: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """
    Deterministically sample n rows from X_train for background data.
    Uses seed to guarantee reproducibility.
    Rejects if n > len(X_train) (would require test data).
    """
    if n > len(X_train):
        raise ValueError(
            f"Requested {n} background rows but only {len(X_train)} training rows available. "
            "Do not use test data as background. Reduce background_n or use more training data."
        )
    rng = np.random.default_rng(seed + 7777)
    idx = rng.choice(len(X_train), size=n, replace=False)
    return X_train[np.sort(idx)]


class KernelSHAPAdapter:
    """
    KernelSHAP explainer adapter. Model-agnostic.

    Background: seeded sample of training rows only.
    Production/confirmatory sampling budgets are not yet approved (DR-003).
    """

    explainer_name = ExplainerName.KERNEL_SHAP
    explainer_type = ExplainerType.SHAP_KERNEL

    def __init__(
        self,
        background_n: int = STAGE3_CI_BACKGROUND_N,
        nsamples: int = STAGE3_CI_NSAMPLES,
    ) -> None:
        self.background_n = background_n
        self.nsamples = nsamples
        self._explainer: shap.KernelExplainer | None = None
        self._background_hash: str | None = None
        self._background_n_rows: int | None = None
        self._background_seed: int | None = None

    def fit(
        self,
        model_predict_proba: Any,  # callable: (X) -> proba_array
        X_train: np.ndarray,
        *,
        seed: int,
    ) -> None:
        """
        Initialise KernelExplainer.

        model_predict_proba: the model's predict_proba method.
        X_train: FULL training partition (background will be sampled from it).
        seed: used for deterministic background sampling.
        """
        background = _sample_background(X_train, self.background_n, seed)
        self._background_hash = ExplanationBatch.hash_background(background)
        self._background_n_rows = len(background)
        self._background_seed = seed

        def _predict_pos_class(X: np.ndarray) -> np.ndarray:
            """Return positive-class probabilities."""
            return model_predict_proba(X)[:, 1]

        self._explainer = shap.KernelExplainer(_predict_pos_class, background)

    def explain(
        self,
        X: np.ndarray,
        *,
        run_id: str,
        dataset: str,
        seed: int,
        model_family: ModelFamily,
        model_hash: str,
        feature_names: list[str],
        sample_ids: list[str],
        protocol_version: str,
        model: Any,
    ) -> ExplanationBatch:
        if self._explainer is None:
            raise RuntimeError("Call fit() first.")

        t0 = time.perf_counter()
        sv = self._explainer.shap_values(X, nsamples=self.nsamples, l1_reg="num_features(10)")
        # KernelExplainer returns array of shape (n, p) for binary
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv)
        base_val = float(self._explainer.expected_value)

        probs = model.predict_proba(X)[:, 1]
        pred_classes = (probs >= 0.5).astype(int).tolist()
        rt = (time.perf_counter() - t0) * 1000

        return ExplanationBatch(
            run_id=run_id,
            protocol_version=protocol_version,
            dataset=dataset,
            seed=seed,
            model_family=model_family,
            model_hash=model_hash,
            explainer=self.explainer_name,
            explainer_type=self.explainer_type,
            explainer_version=shap.__version__,
            explainer_params={
                "background_n": self._background_n_rows,
                "background_seed": self._background_seed,
                "nsamples": self.nsamples,
                "l1_reg": "num_features(10)",
                "output_space": "probability",
            },
            output_space="probability",
            target_class=1,
            sample_ids=sample_ids,
            feature_names=feature_names,
            attributions=sv.tolist(),
            base_values=[base_val] * len(sample_ids),
            predictions=probs.tolist(),
            prediction_classes=pred_classes,
            background_hash=self._background_hash,
            background_n_rows=self._background_n_rows,
            runtime_ms=rt,
            success=True,
            warnings=[],
            failure_reason=None,
        )

    def library_version(self) -> str:
        return shap.__version__

    def explainer_config(self) -> dict[str, Any]:
        return {
            "name": self.explainer_name.value,
            "type": self.explainer_type.value,
            "background_n": self._background_n_rows,
            "background_seed": self._background_seed,
            "background_hash": self._background_hash,
            "nsamples": self.nsamples,
            "shap_version": shap.__version__,
            "note": "Production/confirmatory nsamples budget not yet approved (DR-003).",
        }
