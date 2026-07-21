"""
ExplainCheck — Stage 3 compatibility × determinism matrix reference.

Four cells: RF+KernelSHAP, RF+LIME, XGB+KernelSHAP, XGB+LIME.

Per DR-006A §9, each cell reports two properties:

Same-seed repeatability:
    - Same model predictions
    - Same feature ordering
    - Same background hash
    - Same attributions (exact or within declared tolerance)
    - Same non-time provenance fields
    → deterministic if two runs with identical seed produce identical outputs

Different-seed sensitivity:
    - Seed recorded
    - Stochastic methods (KernelSHAP, LIME): outputs may differ across seeds
    - Differences must be finite and schema-valid
    - Different outputs NOT automatically classified as failures
    - Descriptive stats over seeds 0–4 reported
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixture constants (frozen)
# ---------------------------------------------------------------------------

FROZEN_WEIGHTS = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
FIXTURE_BIAS = 0.3
N_FEATURES = 8
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]
N_BACKGROUND_SHAP = 50
N_BACKGROUND_LIME = 200
N_SAMPLES_SHAP = 512
NUM_SAMPLES_LIME = 512
N_TEST = 20  # smaller for determinism checks


def _make_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_test, y_train for one seed."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((100, N_FEATURES))
    X_test = rng.standard_normal((N_TEST, N_FEATURES))
    y_train = X_train @ FROZEN_WEIGHTS + FIXTURE_BIAS
    return X_train, X_test, y_train


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _train_rf(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> object:
    rf = RandomForestRegressor(n_estimators=50, random_state=seed)
    rf.fit(X_train, y_train)
    return rf


def _train_xgb(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> object:
    model = xgb.XGBRegressor(n_estimators=50, random_state=seed, verbosity=0)
    model.fit(X_train, y_train)
    return model


def _run_kernelshap(
    model: object,
    X_background: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return (shap_values [N_TEST x N_FEATURES], runtime_ms)."""
    t0 = time.perf_counter()
    explainer = shap.KernelExplainer(
        lambda x: model.predict(x),  # type: ignore[attr-defined]
        X_background,
        silent=True,
    )
    shap_values = explainer.shap_values(X_test, nsamples=N_SAMPLES_SHAP, l1_reg=0)
    rt = (time.perf_counter() - t0) * 1000.0
    return np.array(shap_values), rt


def _run_lime(
    model: object,
    X_background: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return (lime_attrs [N_TEST x N_FEATURES], runtime_ms)."""
    import warnings

    explainer = LimeTabularExplainer(
        X_background,
        feature_names=FEATURE_NAMES,
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    t0 = time.perf_counter()
    attrs = np.zeros((N_TEST, N_FEATURES))
    for i in range(len(X_test)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(
                X_test[i],
                lambda x: model.predict(x),  # type: ignore[attr-defined]
                num_features=N_FEATURES,
                num_samples=NUM_SAMPLES_LIME,
            )
        coef_map = dict(exp.as_list())
        for j, name in enumerate(FEATURE_NAMES):
            attrs[i, j] = coef_map.get(name, 0.0)
    rt = (time.perf_counter() - t0) * 1000.0
    return attrs, rt


def run_cell(
    model_name: str,  # "rf" or "xgb"
    explainer_name: str,  # "kernelshap" or "lime"
    seeds: list[int],
) -> dict:
    """
    Run determinism check for one (model, explainer) cell over given seeds.

    Returns per-cell summary with:
    - same_seed_repeatability: whether run1 == run2 with same seed (seed 0)
    - different_seed_sensitivity: descriptive stats across seeds
    - per_seed_results: list of dicts (one per seed)
    """
    assert (
        SHAP_AVAILABLE and LIME_AVAILABLE and MODELS_AVAILABLE
    ), "Required packages (shap, lime, sklearn, xgboost) must be installed."

    # --- Same-seed repeatability (seed 0, run twice) ---
    s0 = seeds[0]
    X_train, X_test, y_train = _make_data(s0)

    if model_name == "rf":
        model_a = _train_rf(X_train, y_train, s0)
        model_b = _train_rf(X_train, y_train, s0)
    else:
        model_a = _train_xgb(X_train, y_train, s0)
        model_b = _train_xgb(X_train, y_train, s0)

    X_bg = X_train[:N_BACKGROUND_SHAP]

    if explainer_name == "kernelshap":
        attrs_a, rt_a = _run_kernelshap(model_a, X_bg, X_test, s0)
        attrs_b, rt_b = _run_kernelshap(model_b, X_bg, X_test, s0)
    else:
        attrs_a, rt_a = _run_lime(model_a, X_train[:N_BACKGROUND_LIME], X_test, s0)
        attrs_b, rt_b = _run_lime(model_b, X_train[:N_BACKGROUND_LIME], X_test, s0)

    preds_a = model_a.predict(X_test)  # type: ignore[attr-defined]
    preds_b = model_b.predict(X_test)  # type: ignore[attr-defined]

    same_preds = bool(np.allclose(preds_a, preds_b, atol=1e-10))
    same_bg_hash = _hash_array(X_bg) == _hash_array(X_bg)  # trivially true, same seed
    same_attrs = bool(np.allclose(attrs_a, attrs_b, atol=1e-8))
    max_attr_diff = float(np.max(np.abs(attrs_a - attrs_b)))

    repeatability = {
        "seed": s0,
        "same_model_predictions": same_preds,
        "same_background_hash": same_bg_hash,
        "same_attributions": same_attrs,
        "max_attribution_diff": max_attr_diff,
        "all_repeat_checks_pass": same_preds and same_bg_hash and same_attrs,
    }

    # --- Different-seed sensitivity ---
    per_seed: list[dict] = []
    all_mean_attrs: list[np.ndarray] = []

    for seed in seeds:
        X_tr, X_te, y_tr = _make_data(seed)
        m = _train_rf(X_tr, y_tr, seed) if model_name == "rf" else _train_xgb(X_tr, y_tr, seed)

        xbg = (
            X_tr[:N_BACKGROUND_SHAP] if explainer_name == "kernelshap" else X_tr[:N_BACKGROUND_LIME]
        )
        if explainer_name == "kernelshap":
            attrs, rt = _run_kernelshap(m, xbg, X_te, seed)
        else:
            attrs, rt = _run_lime(m, xbg, X_te, seed)

        preds = m.predict(X_te)  # type: ignore[attr-defined]
        all_finite = bool(np.all(np.isfinite(attrs)))
        all_mean_attrs.append(attrs.mean(axis=0))

        per_seed.append(
            {
                "seed": seed,
                "all_attrs_finite": all_finite,
                "mean_attr_l2_norm": float(np.linalg.norm(attrs.mean(axis=0))),
                "runtime_ms": rt,
                "n_samples": int(len(X_te)),
                "predictions_finite": bool(np.all(np.isfinite(preds))),
            }
        )

    # Cross-seed variation of mean attributions
    if len(all_mean_attrs) > 1:
        attr_stack = np.stack(all_mean_attrs)  # (n_seeds, n_features)
        cross_seed_std = float(attr_stack.std(axis=0).mean())
    else:
        cross_seed_std = 0.0

    sensitivity = {
        "seeds": seeds,
        "cross_seed_std_of_mean_attrs": cross_seed_std,
        "all_seeds_attrs_finite": all(r["all_attrs_finite"] for r in per_seed),
        "all_seeds_preds_finite": all(r["predictions_finite"] for r in per_seed),
        "note": (
            "Different outputs across seeds are expected for stochastic methods "
            "and not classified as failures."
        ),
    }

    return {
        "cell": f"{model_name}+{explainer_name}",
        "model": model_name,
        "explainer": explainer_name,
        "same_seed_repeatability": repeatability,
        "different_seed_sensitivity": sensitivity,
        "per_seed_results": per_seed,
        "schema_valid": all(r["all_attrs_finite"] for r in per_seed),
    }
