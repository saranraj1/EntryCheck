"""
Stage 2 property-based tests (Hypothesis).

Tests universal mathematical properties that must hold for ALL valid inputs,
not just hand-crafted examples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from explaincheck.metrics.fidelity.aopc import deletion_fidelity_aopc_single
from explaincheck.metrics.stability.top_k_jaccard import jaccard

# ---------------------------------------------------------------------------
# Fidelity properties
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(
    x=st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=10).map(np.array),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=200, deadline=5000)
def test_fidelity_always_nonnegative(x: np.ndarray, seed: int) -> None:
    """Fidelity AOPC must always be >= 0."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=len(x))
    baseline = rng.normal(size=len(x))
    a = rng.normal(size=len(x))
    assume(all(math.isfinite(v) for v in x))
    result = deletion_fidelity_aopc_single(x, a, w, 0.0, baseline, kmax=min(3, len(x)))
    assert result >= 0.0, f"Got {result}"


@pytest.mark.property
@given(
    n=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=100, deadline=5000)
def test_fidelity_baseline_replacement_gives_zero_on_const_logit(n: int, seed: int) -> None:
    """
    If x == baseline for all features, the logit is constant regardless of
    which feature is deleted first → all drops are 0 → AOPC = 0.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    baseline = rng.normal(size=n)
    x = baseline.copy()  # x == baseline
    a = rng.normal(size=n)
    result = deletion_fidelity_aopc_single(x, a, w, 0.0, baseline, kmax=min(3, n))
    assert abs(result) < 1e-12, f"Expected 0.0 when x=baseline, got {result}"


# ---------------------------------------------------------------------------
# Jaccard properties
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(
    a=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=10).map(np.array),
    k=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=300, deadline=2000)
def test_jaccard_reflexive(a: np.ndarray, k: int) -> None:
    """Jaccard(A, A) = 1.0 for any vector (same vector compared with itself)."""
    assume(len(a) >= k)
    result = jaccard(a, a, k)
    assert abs(result - 1.0) < 1e-12, f"Expected 1.0, got {result}"


@pytest.mark.property
@given(
    a=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=10).map(np.array),
    ap=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=10).map(np.array),
    k=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=300, deadline=2000)
def test_jaccard_symmetric(a: np.ndarray, ap: np.ndarray, k: int) -> None:
    """Jaccard(A, A') = Jaccard(A', A)."""
    assume(len(a) == len(ap) and len(a) >= k)
    j1 = jaccard(a, ap, k)
    j2 = jaccard(ap, a, k)
    assert abs(j1 - j2) < 1e-12, f"Jaccard not symmetric: {j1} != {j2}"


@pytest.mark.property
@given(
    a=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=10).map(np.array),
    ap=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=2, max_size=10).map(np.array),
    k=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=300, deadline=2000)
def test_jaccard_range(a: np.ndarray, ap: np.ndarray, k: int) -> None:
    """Jaccard similarity is always in [0, 1]."""
    assume(len(a) == len(ap) and len(a) >= k)
    result = jaccard(a, ap, k)
    assert 0.0 <= result <= 1.0, f"Jaccard out of [0,1]: {result}"


@pytest.mark.property
@given(
    n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
    scale=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=200, deadline=2000)
def test_jaccard_scale_invariant(n: int, seed: int, scale: float) -> None:
    """
    Jaccard top-k is invariant to positive scaling of the attribution vector,
    because it only cares about rank order by absolute value.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    ap = rng.normal(size=n)
    k = min(3, n)
    j1 = jaccard(a, ap, k)
    j2 = jaccard(a * scale, ap * scale, k)
    assert abs(j1 - j2) < 1e-12, f"Jaccard not scale-invariant: {j1} != {j2}"


# ---------------------------------------------------------------------------
# Synthetic generator properties
# ---------------------------------------------------------------------------

@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=50, deadline=10000)
def test_generate_labels_are_binary(seed: int) -> None:
    """All generated labels must be 0 or 1."""
    from explaincheck.datasets.synthetic import generate
    _, y = generate(seed, n=100)
    assert set(y.tolist()).issubset({0, 1})


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=9999))
@settings(max_examples=50, deadline=10000)
def test_generate_no_nan_or_inf(seed: int) -> None:
    """Generated feature matrix must not contain NaN or Inf."""
    from explaincheck.datasets.synthetic import generate
    X, y = generate(seed, n=100)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y.astype(float)))
