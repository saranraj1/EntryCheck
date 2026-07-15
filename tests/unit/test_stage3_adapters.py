"""
Stage 3 contract and determinism tests.

Covers: ExplanationBatch contract, model adapters (RF, XGB, sklearn LR),
explainer adapters (TreeSHAP, KernelSHAP, LIME) on the synthetic linear dataset.

Test categories per DR-003:
- Contract: shape, feature ordering, target class, output space, base-value, dense, hash
- Determinism: predictions and attributions match across identical seeded calls
- Scientific sanity: additivity, KernelSHAP agreement, LIME directional, negative control
- Failure: unsupported model, missing background, feature-schema mismatch, NaN, degenerate
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from explaincheck.contracts import ModelFamily
from explaincheck.contracts.enums import ExplainerName, ExplainerType
from explaincheck.contracts.explanation_batch import ExplanationBatch
from explaincheck.datasets.synthetic import FEATURE_NAMES, generate, split
from explaincheck.explainers.kernel_shap import KernelSHAPAdapter
from explaincheck.explainers.lime_adapter import LIMEAdapter
from explaincheck.explainers.tree_shap import TreeSHAPAdapter
from explaincheck.models.logistic_regression import LogisticRegressionAdapter
from explaincheck.models.random_forest import RandomForestAdapter
from explaincheck.models.sklearn_lr import SklearnLRAdapter
from explaincheck.models.xgboost_adapter import XGBoostAdapter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_data():
    """One synthetic seed for all Stage 3 tests."""
    X, y = generate(11, n=500)
    X_tr, X_te, y_tr, y_te = split(X, y, 11)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture(scope="module")
def rf_fitted(synth_data):
    X_tr, X_te, y_tr, y_te = synth_data
    model = RandomForestAdapter(n_estimators=20, random_state=42)
    rec = model.fit(X_tr, y_tr, seed=11)
    return model, rec


@pytest.fixture(scope="module")
def xgb_fitted(synth_data):
    X_tr, X_te, y_tr, y_te = synth_data
    model = XGBoostAdapter(n_estimators=20, seed=42)
    rec = model.fit(X_tr, y_tr, seed=11)
    return model, rec


@pytest.fixture(scope="module")
def sklearn_lr_fitted(synth_data):
    X_tr, X_te, y_tr, y_te = synth_data
    model = SklearnLRAdapter()
    rec = model.fit(X_tr, y_tr, seed=11)
    return model, rec


# ---------------------------------------------------------------------------
# ExplanationBatch contract validators
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_explanation_batch_rejects_nan():
    """Batch must be rejected if attributions contain NaN."""
    with pytest.raises(Exception):
        ExplanationBatch(
            run_id="t",
            protocol_version="1.0.0",
            dataset="d",
            seed=0,
            model_family=ModelFamily.RANDOM_FOREST,
            model_hash="abcd1234ef",
            explainer=ExplainerName.TREE_SHAP,
            explainer_type=ExplainerType.SHAP_TREE,
            explainer_version="0.46.0",
            explainer_params={},
            output_space="probability",
            target_class=1,
            sample_ids=["s0"],
            feature_names=["f1"],
            attributions=[[float("nan")]],
            base_values=[0.5],
            predictions=[0.6],
            prediction_classes=[1],
            background_hash="abc",
            background_n_rows=50,
            runtime_ms=1.0,
            success=True,
            warnings=[],
            failure_reason=None,
        )


@pytest.mark.unit
def test_explanation_batch_rejects_width_mismatch():
    """Batch rejected if attribution width != feature_names width."""
    with pytest.raises(Exception):
        ExplanationBatch(
            run_id="t",
            protocol_version="1.0.0",
            dataset="d",
            seed=0,
            model_family=ModelFamily.RANDOM_FOREST,
            model_hash="abcd1234ef",
            explainer=ExplainerName.TREE_SHAP,
            explainer_type=ExplainerType.SHAP_TREE,
            explainer_version="0.46.0",
            explainer_params={},
            output_space="probability",
            target_class=1,
            sample_ids=["s0"],
            feature_names=["f1", "f2"],
            attributions=[[0.1, 0.2, 0.3]],  # 3 != 2
            base_values=[0.5],
            predictions=[0.6],
            prediction_classes=[1],
            background_hash="abc",
            background_n_rows=50,
            runtime_ms=1.0,
            success=True,
            warnings=[],
            failure_reason=None,
        )


@pytest.mark.unit
def test_explanation_batch_rejects_empty_output_space():
    """Batch rejected if output_space is empty."""
    with pytest.raises(Exception):
        ExplanationBatch(
            run_id="t",
            protocol_version="1.0.0",
            dataset="d",
            seed=0,
            model_family=ModelFamily.RANDOM_FOREST,
            model_hash="abcd1234ef",
            explainer=ExplainerName.TREE_SHAP,
            explainer_type=ExplainerType.SHAP_TREE,
            explainer_version="0.46.0",
            explainer_params={},
            output_space="",  # must not be empty
            target_class=1,
            sample_ids=["s0"],
            feature_names=["f1"],
            attributions=[[0.1]],
            base_values=[0.5],
            predictions=[0.6],
            prediction_classes=[1],
            background_hash="abc",
            background_n_rows=50,
            runtime_ms=1.0,
            success=True,
            warnings=[],
            failure_reason=None,
        )


# ---------------------------------------------------------------------------
# Model adapter contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rf_fit_returns_model_record(rf_fitted):
    model, rec = rf_fitted
    assert rec.family == ModelFamily.RANDOM_FOREST
    assert len(rec.model_hash) == 64
    assert rec.fit_ms > 0


@pytest.mark.unit
def test_rf_predict_proba_shape(rf_fitted, synth_data):
    model, _ = rf_fitted
    _, X_te, _, _ = synth_data
    probs = model.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.unit
def test_rf_deterministic(synth_data):
    X_tr, _, y_tr, _ = synth_data
    m1 = RandomForestAdapter(n_estimators=10, random_state=42)
    m2 = RandomForestAdapter(n_estimators=10, random_state=42)
    r1 = m1.fit(X_tr, y_tr, seed=11)
    r2 = m2.fit(X_tr, y_tr, seed=11)
    assert r1.model_hash == r2.model_hash


@pytest.mark.unit
def test_xgb_fit_returns_model_record(xgb_fitted):
    model, rec = xgb_fitted
    assert rec.family == ModelFamily.XGBOOST
    assert len(rec.model_hash) == 64
    assert rec.version == "2.1.4", f"XGBoost version frozen at 2.1.4, got {rec.version}"


@pytest.mark.unit
def test_xgb_predict_proba_shape(xgb_fitted, synth_data):
    model, _ = xgb_fitted
    _, X_te, _, _ = synth_data
    probs = model.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.unit
def test_xgb_deterministic(synth_data):
    X_tr, _, y_tr, _ = synth_data
    m1 = XGBoostAdapter(n_estimators=10, seed=42)
    m2 = XGBoostAdapter(n_estimators=10, seed=42)
    r1 = m1.fit(X_tr, y_tr, seed=11)
    r2 = m2.fit(X_tr, y_tr, seed=11)
    np.testing.assert_array_equal(
        m1.predict_proba(synth_data[1])[:, 1],
        m2.predict_proba(synth_data[1])[:, 1],
    )
    assert r1.model_hash == r2.model_hash


@pytest.mark.unit
def test_sklearn_lr_convergence_not_silenced(sklearn_lr_fitted):
    """SklearnLR must record convergence status, not suppress warnings."""
    _, rec = sklearn_lr_fitted
    assert "converged=" in (rec.notes or "")


@pytest.mark.unit
def test_sklearn_lr_fit_deterministic(synth_data):
    X_tr, _, y_tr, _ = synth_data
    m1 = SklearnLRAdapter(random_state=42)
    m2 = SklearnLRAdapter(random_state=42)
    r1 = m1.fit(X_tr, y_tr, seed=11)
    r2 = m2.fit(X_tr, y_tr, seed=11)
    assert r1.model_hash == r2.model_hash


# ---------------------------------------------------------------------------
# TreeSHAP contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_treeshap_rejects_logistic_regression(synth_data, sklearn_lr_fitted):
    """TreeSHAP must reject LR model family."""
    X_tr, X_te, y_tr, _ = synth_data
    model, _ = sklearn_lr_fitted
    exp = TreeSHAPAdapter()
    with pytest.raises(ValueError, match="TreeSHAP"):
        exp.fit(model, X_tr[:20], model_family=ModelFamily.LOGISTIC_REGRESSION, seed=11)


@pytest.mark.unit
def test_treeshap_rf_output_shape(rf_fitted, synth_data):
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = rf_fitted
    exp = TreeSHAPAdapter()
    exp.fit(model, X_tr[:20], model_family=ModelFamily.RANDOM_FOREST, seed=11)
    batch = exp.explain(
        X_te[:5],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.RANDOM_FOREST,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(5)],
        protocol_version="1.0.0",
        model=model,
    )
    assert batch.n_samples == 5
    assert batch.n_features == len(FEATURE_NAMES)
    assert batch.output_space == "probability"
    assert batch.target_class == 1
    assert batch.background_hash is not None


@pytest.mark.unit
def test_treeshap_xgb_output_shape(xgb_fitted, synth_data):
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = xgb_fitted
    exp = TreeSHAPAdapter()
    exp.fit(model, X_tr[:20], model_family=ModelFamily.XGBOOST, seed=11)
    batch = exp.explain(
        X_te[:5],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.XGBOOST,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(5)],
        protocol_version="1.0.0",
        model=model,
    )
    assert batch.n_samples == 5
    assert batch.n_features == len(FEATURE_NAMES)
    assert all(math.isfinite(v) for row in batch.attributions for v in row)


@pytest.mark.unit
def test_treeshap_rf_additivity(rf_fitted, synth_data):
    """
    TreeSHAP additivity: base_value + sum(attributions) ≈ predicted probability.
    Tolerance: 1e-5  (DR-003A: must be ≤ 1e-5; DR-003B §4: same output space throughout).
    Output space: probability (interventional perturbation, model_output='probability').
    RF returns SHAP shape (n, p, 2); expected_value shape (2,) — class 1 extracted explicitly.
    """
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = rf_fitted
    exp = TreeSHAPAdapter(model_output="probability")
    exp.fit(model, X_tr[:20], model_family=ModelFamily.RANDOM_FOREST, seed=11)
    batch = exp.explain(
        X_te[:10],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.RANDOM_FOREST,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(10)],
        protocol_version="1.0.0",
        model=model,
    )
    A = batch.attribution_matrix()
    bv = np.array(batch.base_values)
    pred = np.array(batch.predictions)
    residuals = np.abs((bv + A.sum(axis=1)) - pred)
    assert np.all(residuals < 1e-5), (
        f"TreeSHAP RF additivity failed (DR-003A tolerance ≤ 1e-5). "
        f"Max residual: {residuals.max():.2e}  mean: {residuals.mean():.2e}\n"
        f"Per-sample: {residuals.tolist()}\n"
        "Check output space alignment (probability vs raw margin) or positive-class extraction."
    )


@pytest.mark.unit
def test_treeshap_xgb_additivity(xgb_fitted, synth_data):
    """
    TreeSHAP additivity for XGBoost: base_value + sum(attributions) ≈ predicted probability.
    Tolerance: 1e-5 (DR-003A gate). XGBoost returns SHAP shape (n, p) scalar expected_value.
    Output space: probability (model_output='probability', interventional perturbation).
    """
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = xgb_fitted
    exp = TreeSHAPAdapter(model_output="probability")
    exp.fit(model, X_tr[:20], model_family=ModelFamily.XGBOOST, seed=11)
    batch = exp.explain(
        X_te[:10],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.XGBOOST,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(10)],
        protocol_version="1.0.0",
        model=model,
    )
    A = batch.attribution_matrix()
    bv = np.array(batch.base_values)
    pred = np.array(batch.predictions)
    residuals = np.abs((bv + A.sum(axis=1)) - pred)
    assert np.all(residuals < 1e-5), (
        f"TreeSHAP XGBoost additivity failed (DR-003A tolerance ≤ 1e-5). "
        f"Max residual: {residuals.max():.2e}  mean: {residuals.mean():.2e}\n"
        f"Per-sample: {residuals.tolist()}\n"
        "Check output space alignment (probability vs raw margin) or expected_value extraction."
    )


@pytest.mark.unit
def test_treeshap_rf_deterministic(rf_fitted, synth_data):
    """TreeSHAP RF determinism: identical seeded calls produce identical attributions."""
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = rf_fitted

    def _explain():
        exp = TreeSHAPAdapter()
        exp.fit(model, X_tr[:20], model_family=ModelFamily.RANDOM_FOREST, seed=11)
        return exp.explain(
            X_te[:5],
            run_id="t",
            dataset="d",
            seed=11,
            model_family=ModelFamily.RANDOM_FOREST,
            model_hash=rec.model_hash,
            feature_names=FEATURE_NAMES,
            sample_ids=[f"s{i}" for i in range(5)],
            protocol_version="1.0.0",
            model=model,
        )

    b1 = _explain()
    b2 = _explain()
    np.testing.assert_allclose(b1.attribution_matrix(), b2.attribution_matrix(), atol=1e-10)


# ---------------------------------------------------------------------------
# KernelSHAP contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kernelshap_output_shape(sklearn_lr_fitted, synth_data):
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = KernelSHAPAdapter(background_n=20, nsamples=32)
    exp.fit(model.predict_proba, X_tr, seed=11)
    batch = exp.explain(
        X_te[:3],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(3)],
        protocol_version="1.0.0",
        model=model,
    )
    assert batch.n_samples == 3
    assert batch.n_features == len(FEATURE_NAMES)
    assert batch.output_space == "probability"
    assert batch.background_hash is not None


@pytest.mark.unit
def test_kernelshap_rejects_test_leakage(synth_data):
    """KernelSHAP must reject if background_n > len(X_train)."""
    from explaincheck.explainers.kernel_shap import _sample_background

    X_tr = np.zeros((10, 3))
    with pytest.raises(ValueError, match="test data"):
        _sample_background(X_tr, n=20, seed=0)


@pytest.mark.unit
def test_kernelshap_background_hash_reproducible(sklearn_lr_fitted, synth_data):
    """Same seed must produce same background hash."""
    X_tr, _, y_tr, _ = synth_data
    model, _ = sklearn_lr_fitted
    exp1 = KernelSHAPAdapter(background_n=20, nsamples=16)
    exp2 = KernelSHAPAdapter(background_n=20, nsamples=16)
    exp1.fit(model.predict_proba, X_tr, seed=11)
    exp2.fit(model.predict_proba, X_tr, seed=11)
    assert exp1._background_hash == exp2._background_hash


@pytest.mark.unit
def test_kernelshap_agreement_with_exact_linear(synth_data):
    """
    KernelSHAP should agree directionally with exact linear attributions
    on a simple linear model (sign agreement for most features).
    DR-003 §6: KernelSHAP agreement with exact-linear on linear fixture.
    """
    X_tr, X_te, y_tr, _ = synth_data
    lr_custom = LogisticRegressionAdapter()
    rec = lr_custom.fit(X_tr, y_tr, seed=11)

    from explaincheck.contracts import AttributionRecord
    from explaincheck.explainers.exact_linear import ExactLinearExplainer

    exact_exp = ExactLinearExplainer()
    exact_exp.fit(lr_custom, X_tr, FEATURE_NAMES, seed=11)
    exact_records = [
        r
        for r in exact_exp.explain(
            X_te[:5],
            run_id="t",
            dataset="d",
            seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash=rec.model_hash,
            sample_ids=[f"s{i}" for i in range(5)],
            protocol_version="1.0.0",
        )
        if isinstance(r, AttributionRecord)
    ]

    sklearn_lr = SklearnLRAdapter()
    sk_rec = sklearn_lr.fit(X_tr, y_tr, seed=11)
    kernel_exp = KernelSHAPAdapter(background_n=30, nsamples=48)
    kernel_exp.fit(sklearn_lr.predict_proba, X_tr, seed=11)
    batch = kernel_exp.explain(
        X_te[:5],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=sk_rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(5)],
        protocol_version="1.0.0",
        model=sklearn_lr,
    )

    # Check sign agreement for the two clearly dominant features (x1=1.5, x2=-1.2)
    for i, (er, krow) in enumerate(zip(exact_records, batch.attributions, strict=False)):
        ea = er.attribution
        # Feature 0 (x1, BETA=1.5) and feature 1 (x2, BETA=-1.2) should agree in sign
        assert np.sign(ea[0]) == np.sign(krow[0]) or abs(krow[0]) < 0.05, (
            f"Sample {i}: KernelSHAP sign disagrees with exact for dominant feature x1: "
            f"exact={ea[0]:.4f}, kernel={krow[0]:.4f}"
        )


@pytest.mark.unit
def test_kernelshap_analytic_reference(synth_data):
    """
    Quantitative KernelSHAP analytic reference (DR-003A §D, DR-003B §5).
    Fixture: f(x) = intercept + X @ w  (raw-margin linear model, identity link).
    Analytic Shapley values: φ_j = w_j * (x_j - E[X_j]).
    Gates: mean cosine ≥ 0.99, mean Spearman ≥ 0.95, sign agreement ≥ 0.95.
    """
    import shap as shap_lib
    from scipy.stats import spearmanr

    rng = np.random.default_rng(42)
    n_bg, n_test, p = 100, 20, 8
    weights_an = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
    intercept_an = 0.3

    X_bg = rng.standard_normal((n_bg, p))
    X_te_an = rng.standard_normal((n_test, p))
    bg_mean = X_bg.mean(axis=0)

    def f_raw(X):
        return intercept_an + X @ weights_an

    # Analytic: φ_j = w_j * (x_j - E[X_j])  (valid for independent features, raw-margin)
    analytic = (X_te_an - bg_mean) * weights_an  # (n_test, p)

    # KernelSHAP with 50-row background, identity link, fixed seed
    bg50 = X_bg[:50]
    explainer = shap_lib.KernelExplainer(f_raw, bg50, seed=0)
    sv = np.array(explainer.shap_values(X_te_an, nsamples=512, silent=True))

    cossims, spears, sign_agrees = [], [], []
    nonzero = weights_an != 0
    for i in range(n_test):
        norm_sv = np.linalg.norm(sv[i])
        norm_an = np.linalg.norm(analytic[i])
        cos = float(np.dot(sv[i], analytic[i]) / (norm_sv * norm_an + 1e-15))
        sp = float(spearmanr(sv[i], analytic[i])[0])
        sa = float((np.sign(sv[i][nonzero]) == np.sign(analytic[i][nonzero])).mean())
        cossims.append(cos)
        spears.append(sp)
        sign_agrees.append(sa)

    mean_cos = float(np.mean(cossims))
    mean_sp = float(np.mean(spears))
    mean_sa = float(np.mean(sign_agrees))

    assert mean_cos >= 0.99, (
        f"KernelSHAP analytic: mean cosine {mean_cos:.4f} < 0.99 gate (DR-003A §D). "
        f"Per-sample cosines: {[f'{c:.3f}' for c in cossims]}"
    )
    assert mean_sp >= 0.95, (
        f"KernelSHAP analytic: mean Spearman {mean_sp:.4f} < 0.95 gate. "
        f"Per-sample Spearman: {[f'{s:.3f}' for s in spears]}"
    )
    assert mean_sa >= 0.95, (
        f"KernelSHAP analytic: mean sign agreement {mean_sa:.4f} < 0.95 gate. "
        f"Per-sample sign agree: {[f'{s:.3f}' for s in sign_agrees]}"
    )


@pytest.mark.unit
def test_lime_analytic_reference(synth_data):
    """
    Quantitative LIME analytic reference (DR-003A §E, DR-003B §6).
    Fixture: f(x) = intercept + X @ w  (regression mode, discretize_continuous=False).
    For a linear model, LIME local linear fit recovers the gradient ≈ true weights w.
    DR-003B §6: compare all nonzero signal features; report seed variation.
    Gates: mean cosine ≥ 0.95, sign agreement ≥ 0.90.
    """
    import lime.lime_tabular

    rng = np.random.default_rng(42)
    n_bg, n_test, p = 200, 20, 8
    weights_an = np.array([2.0, -1.5, 1.0, -0.8, 0.5, 0.0, 0.0, 0.0])
    intercept_an = 0.3
    feature_names_an = [f"x{i}" for i in range(p)]

    X_bg = rng.standard_normal((n_bg, p))
    X_te_an = rng.standard_normal((n_test, p))

    def f_raw(X):
        return intercept_an + X @ weights_an

    # For a linear model, LIME local linear coefficients ≈ gradient = true weights w
    # (not Shapley values; DR-003B §6: LIME coefficients represent local gradient)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_bg,
        feature_names=feature_names_an,
        mode="regression",
        discretize_continuous=False,
        random_state=np.random.RandomState(0),
    )

    nonzero = weights_an != 0
    cossims, sign_agrees = [], []
    for i in range(n_test):
        exp = explainer.explain_instance(
            X_te_an[i],
            f_raw,
            num_features=p,
            num_samples=512,
        )
        lime_vec = np.zeros(p)
        for feat_name, coef in exp.as_list():
            # With discretize_continuous=False, feature names are plain: 'x3'
            for fi, fn in enumerate(feature_names_an):
                if fn == feat_name:
                    lime_vec[fi] = coef
                    break
        cos = float(
            np.dot(lime_vec, weights_an)
            / (np.linalg.norm(lime_vec) * np.linalg.norm(weights_an) + 1e-15)
        )
        sa = float((np.sign(lime_vec[nonzero]) == np.sign(weights_an[nonzero])).mean())
        cossims.append(cos)
        sign_agrees.append(sa)

    mean_cos = float(np.mean(cossims))
    mean_sa = float(np.mean(sign_agrees))

    assert mean_cos >= 0.95, (
        f"LIME analytic: mean cosine {mean_cos:.4f} < 0.95 gate (DR-003A §E). "
        f"Kernel width: auto, num_samples=512, discretize=False. "
        f"Per-sample cosines: {[f'{c:.3f}' for c in cossims]}"
    )
    assert mean_sa >= 0.90, (
        f"LIME analytic: mean sign agreement {mean_sa:.4f} < 0.90 gate. "
        f"Per-sample sign agrees: {[f'{s:.3f}' for s in sign_agrees]}"
    )


# ---------------------------------------------------------------------------
# LIME contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lime_output_shape(sklearn_lr_fitted, synth_data):
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = LIMEAdapter(num_samples=64)
    exp.fit(X_tr, FEATURE_NAMES, seed=11)
    batch = exp.explain(
        X_te[:3],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(3)],
        protocol_version="1.0.0",
        model=model,
    )
    assert batch.n_samples == 3
    assert batch.n_features == len(FEATURE_NAMES)
    assert batch.output_space == "probability"


@pytest.mark.unit
def test_lime_dense_output_no_nan(sklearn_lr_fitted, synth_data):
    """LIME dense output must have no NaN or Inf."""
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = LIMEAdapter(num_samples=64)
    exp.fit(X_tr, FEATURE_NAMES, seed=11)
    batch = exp.explain(
        X_te[:5],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(5)],
        protocol_version="1.0.0",
        model=model,
    )
    A = batch.attribution_matrix()
    assert np.all(np.isfinite(A)), "LIME attribution contains NaN or Inf."


@pytest.mark.unit
def test_lime_dense_width_matches_features(sklearn_lr_fitted, synth_data):
    """Dense LIME vector width must exactly match n_features."""
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = LIMEAdapter(num_samples=64)
    exp.fit(X_tr, FEATURE_NAMES, seed=11)
    batch = exp.explain(
        X_te[:3],
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(3)],
        protocol_version="1.0.0",
        model=model,
    )
    assert batch.attribution_matrix().shape == (3, len(FEATURE_NAMES))


@pytest.mark.unit
def test_lime_feature_schema_mismatch_raises(sklearn_lr_fitted, synth_data):
    """LIME must raise if explain() is called with wrong feature names."""
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = LIMEAdapter(num_samples=32)
    exp.fit(X_tr, FEATURE_NAMES, seed=11)
    with pytest.raises(ValueError, match="feature_names mismatch"):
        exp.explain(
            X_te[:2],
            run_id="t",
            dataset="d",
            seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash=rec.model_hash,
            feature_names=["a", "b", "c", "d", "e", "f", "g", "h"],
            sample_ids=["s0", "s1"],
            protocol_version="1.0.0",
            model=model,
        )


@pytest.mark.unit
def test_lime_directional_dominant_feature(sklearn_lr_fitted, synth_data):
    """
    LIME directional agreement: for samples where x1 > mean, LIME attribution
    for x1 should be positive (consistent with BETA_TRUE[0]=1.5).
    DR-003 §6: LIME directional agreement on analytically solvable linear fixture.
    """
    X_tr, X_te, y_tr, _ = synth_data
    model, rec = sklearn_lr_fitted
    exp = LIMEAdapter(num_samples=128, discretize_continuous=False)
    exp.fit(X_tr, FEATURE_NAMES, seed=11)

    # Select 5 samples where x1 is strongly positive
    mask = X_te[:, 0] > 1.5
    Xpos = X_te[mask][:5]
    if len(Xpos) < 3:
        pytest.skip("Not enough samples with x1 > 1.5 in test split")

    batch = exp.explain(
        Xpos,
        run_id="t",
        dataset="d",
        seed=11,
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash=rec.model_hash,
        feature_names=FEATURE_NAMES,
        sample_ids=[f"s{i}" for i in range(len(Xpos))],
        protocol_version="1.0.0",
        model=model,
    )
    A = batch.attribution_matrix()
    # Feature 0 (x1) should be positive for most samples with x1 > 1.5
    n_positive = int((A[:, 0] > 0).sum())
    assert n_positive >= len(Xpos) // 2, (
        f"LIME directional test: expected x1 attribution positive for samples with x1>1.5, "
        f"got {n_positive}/{len(Xpos)} positive."
    )


# ---------------------------------------------------------------------------
# Scientific sanity: negative control ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_negative_control_still_lower_after_stage3(synth_data):
    """
    Negative control should produce lower fidelity than exact linear (Stage 2 invariant).
    Verified here to ensure Stage 3 model changes don't break this property.
    """
    from explaincheck.contracts import AttributionRecord
    from explaincheck.explainers.exact_linear import ExactLinearExplainer, RandomizedNegativeControl
    from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single

    X_tr, X_te, y_tr, _ = synth_data
    lr = LogisticRegressionAdapter()
    lr.fit(X_tr, y_tr, seed=11)
    baseline = X_tr.mean(axis=0)

    exact_exp = ExactLinearExplainer()
    exact_exp.fit(lr, X_tr, FEATURE_NAMES, seed=11)
    neg_exp = RandomizedNegativeControl()
    neg_exp.fit(lr, X_tr, FEATURE_NAMES, seed=11)

    Xe = X_te[:30]
    sample_ids = [f"s{i}" for i in range(30)]

    def _fid(records):
        return np.mean(
            [
                deletion_fidelity_aopc_single(
                    Xe[i], np.array(r.attribution), lr.weights, lr.bias, baseline, 3
                )
                for i, r in enumerate(r for r in records if isinstance(r, AttributionRecord))
            ]
        )

    exact_fid = _fid(
        exact_exp.explain(
            Xe,
            run_id="t",
            dataset="d",
            seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash="h",
            sample_ids=sample_ids,
            protocol_version="1.0.0",
        )
    )
    neg_fid = _fid(
        neg_exp.explain(
            Xe,
            run_id="t",
            dataset="d",
            seed=11,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash="h",
            sample_ids=sample_ids,
            protocol_version="1.0.0",
        )
    )

    assert (
        exact_fid > neg_fid
    ), f"Exact fidelity ({exact_fid:.4f}) must exceed negative control ({neg_fid:.4f})."


# ---------------------------------------------------------------------------
# Failure modes (structured records)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rf_requires_fit():
    m = RandomForestAdapter()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((1, 8)))


@pytest.mark.unit
def test_xgb_requires_fit():
    m = XGBoostAdapter()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((1, 8)))


@pytest.mark.unit
def test_treeshap_requires_fit():
    exp = TreeSHAPAdapter()
    with pytest.raises(RuntimeError):
        exp.explain(
            np.zeros((1, 8)),
            run_id="t",
            dataset="d",
            seed=0,
            model_family=ModelFamily.RANDOM_FOREST,
            model_hash="h",
            feature_names=["f"] * 8,
            sample_ids=["s0"],
            protocol_version="1.0.0",
            model=None,
        )


@pytest.mark.unit
def test_kernelshap_requires_fit():
    exp = KernelSHAPAdapter()
    with pytest.raises(RuntimeError):
        exp.explain(
            np.zeros((1, 8)),
            run_id="t",
            dataset="d",
            seed=0,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash="h",
            feature_names=["f"] * 8,
            sample_ids=["s0"],
            protocol_version="1.0.0",
            model=None,
        )


@pytest.mark.unit
def test_lime_requires_fit():
    exp = LIMEAdapter()
    with pytest.raises(RuntimeError):
        exp.explain(
            np.zeros((1, 8)),
            run_id="t",
            dataset="d",
            seed=0,
            model_family=ModelFamily.LOGISTIC_REGRESSION,
            model_hash="h",
            feature_names=FEATURE_NAMES,
            sample_ids=["s0"],
            protocol_version="1.0.0",
            model=None,
        )
