"""
ExplainCheck test suite — shared fixtures and configuration.

Marker guide:
    @pytest.mark.unit        — fast, isolated, no I/O
    @pytest.mark.property    — Hypothesis property tests
    @pytest.mark.integration — requires installed packages, may be slow
    @pytest.mark.golden      — frozen fixture comparison
    @pytest.mark.scientific  — validates scientific correctness against analytic reference
    @pytest.mark.smoke       — quick infrastructure check
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared numeric fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests — always seed 42."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_linear_weights() -> np.ndarray:
    """Weights from the Phase 0 manual validation fixture."""
    return np.array([1.0, 0.5, 0.2])


@pytest.fixture
def simple_linear_sample() -> np.ndarray:
    """Sample from the Phase 0 manual validation fixture."""
    return np.array([2.0, 1.0, 1.0])


@pytest.fixture
def simple_linear_baseline() -> np.ndarray:
    """Baseline from the Phase 0 manual validation fixture."""
    return np.zeros(3)


@pytest.fixture
def simple_linear_attribution() -> np.ndarray:
    """
    Expected attribution from the Phase 0 manual validation fixture.
    exact_linear: weight * (x - baseline) = [1.0*2.0, 0.5*1.0, 0.2*1.0]
    """
    return np.array([2.0, 0.5, 0.2])


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_10samples(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """10-sample synthetic dataset for quick unit tests."""
    X = rng.normal(size=(10, 8))
    w = np.array([1.5, -1.2, 0.9, -0.7, 0.0, 0.0, 0.0, 0.0])
    logits = X @ w - 0.15
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
    y = (p >= 0.5).astype(int)
    return X, y
