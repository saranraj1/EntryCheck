"""
ExplainCheck configuration loader.

Loads and validates YAML config files against the protocol schema.
Config objects are immutable Pydantic models once loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    generator: str = "linear_logit"
    n_per_seed: int = 3000
    n_features: int = 8
    true_coefficients: list[float]
    intercept: float = -0.15
    feature_names: list[str]
    train_fraction: float = 0.80


class ReproductionTolerance(BaseModel):
    model_config = ConfigDict(frozen=True)

    roc_auc_mean: float | None = None
    roc_auc_tolerance: float | None = None
    fidelity_exact_n200: float | None = None
    fidelity_tolerance: float | None = None
    stability_exact_n200: float | None = None
    stability_tolerance: float | None = None


class ExperimentConfig(BaseModel):
    """Fully validated experiment configuration loaded from a YAML file."""

    model_config = ConfigDict(frozen=True)

    run_label: str
    status: str
    seeds: list[int]
    protocol_version: str = "1.0.0"
    study_id: str = "EC-TABULAR-001"
    dataset_mode: str = "real"       # "real" or "synthetic"
    synthetic: SyntheticConfig | None = None
    output_dir: str = "artifacts/pilot"
    overwrite_protection: bool = False
    reproduction_tolerance: ReproductionTolerance | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> tuple[ExperimentConfig, str]:
    """
    Load and validate a YAML config. Returns (config, sha256_hex).

    The SHA-256 is computed over the raw file bytes so it can be
    embedded in run manifests for reproducibility.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    config_hash = hashlib.sha256(raw_bytes).hexdigest()
    data: dict[str, Any] = yaml.safe_load(raw_bytes.decode("utf-8")) or {}

    # Pull top-level fields into validated model
    cfg = ExperimentConfig(
        run_label=data.get("run_label", "unknown"),
        status=data.get("status", "unknown"),
        seeds=data.get("seeds", []),
        protocol_version=data.get("protocol_version", "1.0.0"),
        study_id=data.get("study_id", "EC-TABULAR-001"),
        dataset_mode=data.get("dataset_mode", "real"),
        synthetic=_parse_synthetic(data.get("synthetic")),
        output_dir=data.get("output_dir", "artifacts/pilot"),
        overwrite_protection=data.get("overwrite_protection", False),
        reproduction_tolerance=_parse_tolerance(data.get("reproduction_tolerance")),
        raw=data,
    )
    return cfg, config_hash


def _parse_synthetic(raw: dict[str, Any] | None) -> SyntheticConfig | None:
    if raw is None:
        return None
    return SyntheticConfig(**raw)


def _parse_tolerance(raw: dict[str, Any] | None) -> ReproductionTolerance | None:
    if raw is None:
        return None
    return ReproductionTolerance(**raw)
