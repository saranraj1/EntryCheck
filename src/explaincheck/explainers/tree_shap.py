"""
ExplainCheck — TreeSHAP explainer adapter (Stage 3).

DR-003 §3 requirements:
- Compatible with Random Forest and XGBoost only
- Explain positive-class probability (output_space="probability") or
  precisely documented raw_margin — never mix output spaces
- Record model_output, perturbation mode, background hash, target class, SHAP version
- Test SHAP additivity in the selected output space
- Return common ExplanationBatch contract
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import shap

from explaincheck.contracts import ExplainerName, ExplainerType, ModelFamily
from explaincheck.contracts.explanation_batch import ExplanationBatch
from explaincheck.models.random_forest import RandomForestAdapter
from explaincheck.models.xgboost_adapter import XGBoostAdapter

# Compatible model families for TreeSHAP
TREESHAP_COMPATIBLE = {ModelFamily.RANDOM_FOREST, ModelFamily.XGBOOST}


def _require_tree_model(model_family: ModelFamily) -> None:
    if model_family not in TREESHAP_COMPATIBLE:
        raise ValueError(
            f"TreeSHAP is only compatible with {[f.value for f in TREESHAP_COMPATIBLE]}. "
            f"Got: {model_family.value}. "
            "Use KernelSHAP for model-agnostic explanations."
        )


class TreeSHAPAdapter:
    """
    TreeSHAP explainer adapter.

    Output space: probability (positive class). Records all DR-003 provenance fields.
    Additivity: base_value + sum(attributions) ≈ model logit (not probability).
    Note: SHAP additivity holds in the logit/margin space, not probability.
    """

    explainer_name = ExplainerName.TREE_SHAP
    explainer_type = ExplainerType.SHAP_TREE

    def __init__(self, model_output: str = "probability") -> None:
        """
        model_output: "probability" (default) or "raw" (log-odds/margin).
        DR-003: must document and must not mix across models.
        """
        self.model_output = model_output
        self._explainer: shap.TreeExplainer | None = None
        self._background_hash: str | None = None
        self._background_n_rows: int | None = None
        self._model_family: ModelFamily | None = None

    def fit(
        self,
        model: RandomForestAdapter | XGBoostAdapter,
        X_background: np.ndarray,
        *,
        model_family: ModelFamily,
        seed: int,
    ) -> None:
        """
        Initialise TreeExplainer.

        X_background: a small sample from the TRAINING partition only.
        model_family: must be RF or XGB.
        """
        _require_tree_model(model_family)
        self._model_family = model_family
        self._background_hash = ExplanationBatch.hash_background(X_background)
        self._background_n_rows = len(X_background)

        sklearn_model = model.sklearn_model
        self._explainer = shap.TreeExplainer(
            sklearn_model,
            data=X_background,
            model_output=self.model_output,
            feature_perturbation="interventional",
        )

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
        model: RandomForestAdapter | XGBoostAdapter,
    ) -> ExplanationBatch:
        if self._explainer is None or self._model_family is None:
            raise RuntimeError("Call fit() first.")

        _require_tree_model(model_family)

        t0 = time.perf_counter()
        shap_values = self._explainer.shap_values(X)
        base_vals_raw = self._explainer.expected_value

        # DR-003B §4: positive-class extraction must be explicit, not inferred from dimensions alone.
        #
        # SHAP output shapes observed in practice for binary classification:
        #   RF  + model_output="probability"  -> ndarray (n, p, 2), expected_value shape (2,)
        #   XGB + model_output="probability"  -> ndarray (n, p),    expected_value scalar float
        #   older SHAP / some configs          -> list [class0_arr, class1_arr]
        #
        # In all cases we extract the positive class (class 1) explicitly.
        if isinstance(shap_values, list):
            # Old SHAP convention: list [class0_arr, class1_arr]
            sv = np.array(shap_values[1])
            bv = (
                float(base_vals_raw[1])
                if hasattr(base_vals_raw, "__len__")
                else float(base_vals_raw)
            )
        elif shap_values.ndim == 3 and shap_values.shape[2] == 2:
            # RF: (n_samples, n_features, n_classes) — take class 1 slice
            sv = shap_values[:, :, 1]
            bv = (
                float(base_vals_raw[1])
                if hasattr(base_vals_raw, "__len__") and len(base_vals_raw) == 2
                else float(base_vals_raw)
            )
        elif shap_values.ndim == 2:
            # XGB: (n_samples, n_features) with scalar expected_value for positive class
            sv = shap_values
            bv = (
                float(base_vals_raw)
                if not hasattr(base_vals_raw, "__len__")
                else float(np.asarray(base_vals_raw).flat[0])
            )
        else:
            raise ValueError(
                f"Unexpected SHAP values shape {shap_values.shape}. "
                "Supported: list[arr,arr], ndarray(n,p,2), ndarray(n,p). "
                "Please file a bug report with your SHAP version and model type."
            )

        probs = model.predict_proba(X)[:, 1]
        pred_classes = (probs >= 0.5).astype(int).tolist()
        rt = (time.perf_counter() - t0) * 1000

        base_values_list = [bv] * len(sample_ids)

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
                "model_output": self.model_output,
                "feature_perturbation": "interventional",
                "background_hash": self._background_hash,
                "background_n_rows": self._background_n_rows,
            },
            output_space=self.model_output,
            target_class=1,
            sample_ids=sample_ids,
            feature_names=feature_names,
            attributions=sv.tolist(),
            base_values=base_values_list,
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
            "model_output": self.model_output,
            "feature_perturbation": "interventional",
            "background_hash": self._background_hash,
            "background_n_rows": self._background_n_rows,
            "shap_version": shap.__version__,
            "compatible_models": [f.value for f in TREESHAP_COMPATIBLE],
        }
