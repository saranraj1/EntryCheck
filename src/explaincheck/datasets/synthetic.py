"""
ExplainCheck — Synthetic linear-logit dataset generator.

Migrated from Phase 0 (run_phase0.py) without changing scientific definitions.
Generates binary classification data from a known logistic model for analytic validation.

Scientific definition (frozen from Phase 0):
    y ~ Bernoulli(sigmoid(X @ beta_true + intercept))
    X ~ N(0, I)  (independent standard Gaussian features)
    Train/test split: 80/20, seed-specific deterministic permutation.
"""

from __future__ import annotations

import numpy as np

from explaincheck.contracts import DataSplit, DatasetRecord, SplitRecord, TaskType
from explaincheck.provenance import hash_array, hash_string, utc_now_iso

# Frozen from Phase 0 / pilot.yaml
BETA_TRUE = np.array([1.50, -1.20, 0.90, -0.70, 0.0, 0.0, 0.0, 0.0])
FEATURE_NAMES = [f"x{i + 1}" for i in range(len(BETA_TRUE))]
INTERCEPT = -0.15
N_PER_SEED = 3000
TRAIN_FRACTION = 0.80


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. Identical to Phase 0."""
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def generate(
    seed: int,
    n: int = N_PER_SEED,
    beta: np.ndarray | None = None,
    intercept: float = INTERCEPT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n samples from the synthetic linear-logit model.

    Parameters
    ----------
    seed : int
        Random seed (must be one of the ten frozen seeds in confirmatory runs).
    n : int
        Number of samples to generate.
    beta : array-like, optional
        True coefficients. Defaults to BETA_TRUE.
    intercept : float
        Logit intercept. Defaults to INTERCEPT.

    Returns
    -------
    X : (n, p) float64 array
    y : (n,) int64 array of binary labels
    """
    if beta is None:
        beta = BETA_TRUE
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    logits = X @ beta + intercept
    probs = _sigmoid(logits)
    y = rng.binomial(1, probs).astype(np.int64)
    return X, y


def split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic train/test split.

    Uses seed + 1000 for the permutation (identical to Phase 0) to guarantee
    that the split RNG is independent of the data-generation RNG.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(seed + 1000)
    idx = rng.permutation(len(y))
    cut = int(train_fraction * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def split_record(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    dataset_name: str = "synthetic_linear",
) -> SplitRecord:
    """Return a SplitRecord with SHA-256 hashes of each half."""
    import hashlib

    def _hash(X: np.ndarray, y: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(X.tobytes())
        h.update(y.tobytes())
        return h.hexdigest()

    return SplitRecord(
        dataset=dataset_name,
        seed=seed,
        train_sha256=_hash(X_train, y_train),
        test_sha256=_hash(X_test, y_test),
        train_n=len(y_train),
        test_n=len(y_test),
    )


def dataset_record(
    seed: int,
    n: int = N_PER_SEED,
    beta: np.ndarray | None = None,
) -> DatasetRecord:
    """Return a DatasetRecord for the synthetic dataset."""
    if beta is None:
        beta = BETA_TRUE
    config_str = f"seed={seed},n={n},beta={beta.tolist()},intercept={INTERCEPT}"
    return DatasetRecord(
        name="synthetic_linear",
        version="1.0.0-phase0",
        doi="N/A",
        url="N/A",
        license="N/A (synthetic)",
        retrieval_date=utc_now_iso()[:10],
        sha256=hash_string(config_str),
        n_rows=n,
        n_features=len(beta),
        target_column="y",
        task=TaskType.BINARY_CLASSIFICATION,
        notes="Synthetic linear-logit dataset. Migrated from Phase 0. Not a real-world dataset.",
    )
