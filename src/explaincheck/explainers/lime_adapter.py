"""
ExplainCheck — LIME explainer adapter (Stage 3).

DR-003 §3 requirements:
- LimeTabularExplainer
- Pass random seed explicitly
- Fit from training data only
- Return dense attribution vector aligned with feature schema
- Zero-fill features absent from the sparse LIME output
- Record kernel_width, discretize_continuous, num_samples, target_class, feature_mapping
- Keep categorical feature handling explicit
- Return common ExplanationBatch contract
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from explaincheck.contracts import ExplainerName, ExplainerType, ModelFamily
from explaincheck.contracts.explanation_batch import ExplanationBatch

try:
    from importlib.metadata import version as _pkg_version

    import lime
    import lime.lime_tabular

    try:
        LIME_VERSION = _pkg_version("lime")
    except Exception:
        LIME_VERSION = getattr(lime, "__version__", "unknown")
except ImportError:
    LIME_VERSION = "not-installed"


class LIMEAdapter:
    """
    LIME tabular explainer adapter.

    Dense output: features absent from LIME's sparse explanation are zeroed
    (not silently discarded) per DR-003.
    """

    explainer_name = ExplainerName.LIME
    explainer_type = ExplainerType.LIME

    def __init__(
        self,
        num_samples: int = 512,
        kernel_width: float | None = None,
        discretize_continuous: bool = True,
    ) -> None:
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous
        self._explainer: Any | None = None
        self._feature_names: list[str] | None = None
        self._background_hash: str | None = None
        self._background_n_rows: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
        *,
        seed: int,
        categorical_features: list[int] | None = None,
    ) -> None:
        """
        Initialise LimeTabularExplainer from training data.

        categorical_features: list of column indices that are categorical.
        """
        from explaincheck.contracts.explanation_batch import ExplanationBatch

        self._feature_names = list(feature_names)
        self._background_hash = ExplanationBatch.hash_background(X_train)
        self._background_n_rows = len(X_train)

        kw: dict[str, Any] = {
            "training_data": X_train,
            "feature_names": feature_names,
            "mode": "classification",
            "discretize_continuous": self.discretize_continuous,
            "random_state": seed,
        }
        if categorical_features is not None:
            kw["categorical_features"] = categorical_features
        if self.kernel_width is not None:
            kw["kernel_width"] = self.kernel_width

        self._explainer = lime.lime_tabular.LimeTabularExplainer(**kw)

    def _explain_one(
        self,
        x: np.ndarray,
        predict_fn: Any,
        seed: int,
        target_class: int,
    ) -> tuple[np.ndarray, float]:
        """
        Explain one sample. Returns (dense_attribution, local_prediction).
        Absent LIME features are zero-filled.
        """
        assert self._explainer is not None and self._feature_names is not None
        n_features = len(self._feature_names)

        exp = self._explainer.explain_instance(
            x,
            predict_fn,
            num_features=n_features,
            num_samples=self.num_samples,
            labels=(target_class,),
        )

        # Build dense attribution vector (zero for missing features)
        dense = np.zeros(n_features)
        feature_map = {name: i for i, name in enumerate(self._feature_names)}
        for feat_name, weight in exp.as_list(label=target_class):
            # LIME returns feature names that may have discretization suffixes
            # Find which original feature this refers to (exact match first, then prefix)
            if feat_name in feature_map:
                idx = feature_map[feat_name]
                dense[idx] = weight
            else:
                # Try to match by prefix (discretized features have ranges appended)
                for orig_name, idx in feature_map.items():
                    if feat_name.startswith(orig_name):
                        dense[idx] = weight
                        break

        # local_pred is a dict {label: local_prediction} in this version of lime
        if hasattr(exp, "local_pred") and exp.local_pred is not None:
            lp = exp.local_pred
            if isinstance(lp, dict):
                local_pred = float(lp.get(target_class, float("nan")))
            else:
                # Fallback: older LIME stores as array indexed by labels tuple position
                try:
                    local_pred = float(lp[target_class])
                except (IndexError, KeyError):
                    local_pred = float("nan")
        else:
            local_pred = float("nan")
        return dense, local_pred

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
        target_class: int = 1,
    ) -> ExplanationBatch:
        if self._explainer is None or self._feature_names is None:
            raise RuntimeError("Call fit() first.")

        if feature_names != self._feature_names:
            raise ValueError(
                "feature_names mismatch: explainer was fitted with different feature names. "
                "Batch rejected per DR-003 §4 (feature-schema mismatch)."
            )

        def _predict_fn(X_arr: np.ndarray) -> np.ndarray:
            return model.predict_proba(X_arr)

        t0 = time.perf_counter()
        all_attrs: list[list[float]] = []
        all_local_preds: list[float] = []
        rng = np.random.default_rng(seed + 3333)

        for i in range(len(X)):
            # LIME's random_state must be a np.random.RandomState (not a plain int).
            # DR-003: pass seed explicitly to every call.
            sample_seed = int(rng.integers(0, 2**31))
            self._explainer.random_state = np.random.RandomState(sample_seed)
            dense, local_pred = self._explain_one(X[i], _predict_fn, sample_seed, target_class)
            all_attrs.append(dense.tolist())
            all_local_preds.append(float(local_pred))

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
            explainer_version=LIME_VERSION,
            explainer_params={
                "num_samples": self.num_samples,
                "kernel_width": self.kernel_width,
                "discretize_continuous": self.discretize_continuous,
                "target_class": target_class,
                "background_n_rows": self._background_n_rows,
                "background_hash": self._background_hash,
                "dense_output": True,
                "zero_fill_missing": True,
            },
            output_space="probability",
            target_class=target_class,
            sample_ids=sample_ids,
            feature_names=feature_names,
            attributions=all_attrs,
            base_values=None,  # LIME does not have a global base value
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
        return LIME_VERSION

    def explainer_config(self) -> dict[str, Any]:
        return {
            "name": self.explainer_name.value,
            "type": self.explainer_type.value,
            "num_samples": self.num_samples,
            "kernel_width": self.kernel_width,
            "discretize_continuous": self.discretize_continuous,
            "dense_output": True,
            "zero_fill_missing": True,
            "background_hash": self._background_hash,
            "background_n_rows": self._background_n_rows,
            "lime_version": LIME_VERSION,
        }
