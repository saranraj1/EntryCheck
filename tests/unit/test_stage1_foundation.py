"""
Unit tests — Stage 1 structural validation.

These tests verify the package structure, import paths, enum values,
Pydantic model validation, provenance hashing, and config loading.
All tests are fast (< 1 second each) and require no external packages
beyond the core install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import explaincheck
from explaincheck.contracts import (
    ExplainerName,
    MetricFamily,
    ModelFamily,
    RunStatus,
)
from explaincheck.contracts.models import (
    DatasetRecord,
    FailureRecord,
    MetricResult,
)
from explaincheck.provenance import (
    hash_bytes,
    hash_json,
    hash_string,
    new_run_id,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_package_version_defined() -> None:
    assert explaincheck.__version__ == "0.1.0-dev"


@pytest.mark.unit
def test_study_id_correct() -> None:
    assert explaincheck.__study_id__ == "EC-TABULAR-001"


@pytest.mark.unit
def test_protocol_version_correct() -> None:
    assert explaincheck.__protocol_version__ == "1.0.0"


# ---------------------------------------------------------------------------
# Enum correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_run_status_values() -> None:
    assert RunStatus.SUCCESS == "success"
    assert RunStatus.FAILURE == "failure"
    assert RunStatus.EXCLUDED == "excluded"


@pytest.mark.unit
def test_model_family_xgboost_present() -> None:
    assert ModelFamily.XGBOOST == "xgboost"


@pytest.mark.unit
def test_explainer_names_complete() -> None:
    names = {e.value for e in ExplainerName}
    assert "tree_shap" in names
    assert "kernel_shap" in names
    assert "lime" in names
    assert "exact_linear" in names
    assert "randomized_negative_control" in names


@pytest.mark.unit
def test_metric_family_seven_dimensions() -> None:
    """H1 wording correction: seven evaluation dimensions must all be present."""
    families = {f.value for f in MetricFamily}
    required = {
        "fidelity", "stability", "sparsity",
        "runtime", "correlation", "missingness", "subgroup"
    }
    assert required == families, (
        f"MetricFamily must contain exactly 7 families. Missing: {required - families}"
    )


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dataset_record_immutable() -> None:
    record = DatasetRecord(
        name="test",
        version="1.0",
        doi="10.0000/test",
        url="https://example.com",
        license="CC BY 4.0",
        retrieval_date="2026-07-14",
        sha256="abc123",
        n_rows=100,
        n_features=5,
        target_column="y",
        task="binary_classification",
    )
    with pytest.raises(Exception):
        record.name = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_failure_record_requires_reason() -> None:
    """FailureRecord must always carry a failure_reason."""
    with pytest.raises(Exception):
        FailureRecord(  # type: ignore[call-arg]
            run_id="ec-test",
            timestamp="2026-07-14T00:00:00Z",
            dataset="test",
            seed=11,
            # Missing failure_reason — should fail validation
        )


@pytest.mark.unit
def test_metric_result_schema_version() -> None:
    result = MetricResult(
        run_id="ec-test",
        protocol_version="1.0.0",
        dataset="test",
        dataset_version="1.0",
        split_hash="abc",
        model_family=ModelFamily.LOGISTIC_REGRESSION,
        model_hash="def",
        explainer=ExplainerName.EXACT_LINEAR,
        explainer_version="0.1",
        seed=11,
        metric_family=MetricFamily.FIDELITY,
        metric_name="deletion_fidelity_aopc",
        prediction_preservation_status="not_applicable",
        estimate=1.764,
        runtime_ms=12.5,
        status=RunStatus.SUCCESS,
    )
    assert result.schema_version == "1.0.0"
    assert result.estimate == 1.764


# ---------------------------------------------------------------------------
# Provenance hashing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hash_string_deterministic() -> None:
    h1 = hash_string("explaincheck")
    h2 = hash_string("explaincheck")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


@pytest.mark.unit
def test_hash_bytes_known_value() -> None:
    # SHA-256 of b"" is well-known
    import hashlib
    expected = hashlib.sha256(b"").hexdigest()
    assert hash_bytes(b"") == expected


@pytest.mark.unit
def test_hash_json_key_order_invariant() -> None:
    h1 = hash_json({"b": 2, "a": 1})
    h2 = hash_json({"a": 1, "b": 2})
    assert h1 == h2  # sorted keys → same hash


@pytest.mark.unit
def test_new_run_id_format() -> None:
    rid = new_run_id()
    assert rid.startswith("ec-")
    assert len(rid) == 11  # "ec-" + 8 hex chars


@pytest.mark.unit
def test_new_run_id_unique() -> None:
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100  # no collisions


@pytest.mark.unit
def test_utc_now_iso_format() -> None:
    ts = utc_now_iso()
    from datetime import datetime
    # Must parse as UTC ISO-8601
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_pilot_config(tmp_path: Path) -> None:
    from explaincheck.config import load_config

    config_text = """
run_label: pilot
status: infrastructure-pilot
seeds: [11, 23]
dataset_mode: synthetic
output_dir: artifacts/pilot
"""
    cfg_file = tmp_path / "test_pilot.yaml"
    cfg_file.write_text(config_text, encoding="utf-8")
    cfg, cfg_hash = load_config(cfg_file)

    assert cfg.run_label == "pilot"
    assert cfg.seeds == [11, 23]
    assert len(cfg_hash) == 64  # SHA-256 hex


@pytest.mark.unit
def test_config_hash_changes_with_content(tmp_path: Path) -> None:
    from explaincheck.config import load_config

    base = "run_label: smoke\nstatus: infrastructure-only\nseeds: [11]\n"
    modified = base + "# comment added\n"

    f1 = tmp_path / "a.yaml"; f1.write_text(base, encoding="utf-8")
    f2 = tmp_path / "b.yaml"; f2.write_text(modified, encoding="utf-8")

    _, h1 = load_config(f1)
    _, h2 = load_config(f2)
    assert h1 != h2
