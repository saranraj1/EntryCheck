"""
Stage 2 CLI integration tests.

Verifies the CLI commands work end-to-end without errors.
Uses --dry-run for fast checks; slow tests run the actual pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from explaincheck.cli.main import cli


@pytest.mark.unit
def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0-dev" in result.output


@pytest.mark.unit
def test_cli_env_snapshot_stdout() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["env-snapshot"])
    assert result.exit_code == 0
    env = json.loads(result.output)
    assert "python_version" in env
    assert "xgboost_version" in env
    assert env["xgboost_version"] == "2.1.4"


@pytest.mark.unit
def test_cli_env_snapshot_to_file(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "env.json"
    result = runner.invoke(cli, ["env-snapshot", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    env = json.loads(out.read_text(encoding="utf-8"))
    assert env["xgboost_version"] == "2.1.4"


@pytest.mark.unit
def test_cli_pilot_synthetic_dry_run(tmp_path: Path) -> None:
    """Dry run must succeed without running any computation."""
    runner = CliRunner()
    import os, shutil
    # Copy the pilot config to a temp location to avoid path issues
    src = Path("configs/pilot.yaml")
    dst = tmp_path / "pilot.yaml"
    shutil.copy(src, dst)
    result = runner.invoke(cli, ["pilot", "synthetic", "--config", str(dst), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "Config valid" in result.output


@pytest.mark.unit
def test_cli_smoke_config_invalid_status(tmp_path: Path) -> None:
    """A confirmatory config must be rejected by the pilot command."""
    runner = CliRunner()
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("run_label: confirmatory\nstatus: confirmatory\nseeds: [11]\n", encoding="utf-8")
    result = runner.invoke(cli, ["pilot", "synthetic", "--config", str(cfg), "--dry-run"])
    assert result.exit_code != 0 or "ERROR" in result.output


@pytest.mark.unit
def test_cli_validate_artifacts_missing_manifest(tmp_path: Path) -> None:
    """validate-artifacts must fail if run-manifest.json is absent."""
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-artifacts", "--dir", str(tmp_path)])
    assert result.exit_code != 0


@pytest.mark.integration
@pytest.mark.slow
def test_cli_pilot_full_run(tmp_path: Path) -> None:
    """
    Full pilot run via CLI — integration test.

    Runs two seeds only for speed. Verifies all required artifacts are created.
    """
    runner = CliRunner()
    # Write a 2-seed pilot config
    cfg = tmp_path / "pilot2.yaml"
    cfg.write_text(
        "run_label: pilot\nstatus: infrastructure-pilot\n"
        "seeds: [11, 23]\ndataset_mode: synthetic\n"
        f"output_dir: {tmp_path / 'stage2-cli-test'}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "stage2-cli-test"

    result = runner.invoke(
        cli,
        ["pilot", "synthetic", "--config", str(cfg), "--out", str(out_dir)],
    )

    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert "DONE" in result.output
    assert "infrastructure-pilot" in result.output.lower()

    # Verify required artifacts exist
    required = [
        "run-manifest.json", "environment.json", "benchmark.json",
        "tidy-results.csv", "model-performance.csv", "failures.csv",
        "manual-metric-validation.json", "model-card.md", "SHA256SUMS.txt",
    ]
    for f in required:
        assert (out_dir / f).exists(), f"Missing: {f}"

    manifest = json.loads((out_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["runLabel"] == "infrastructure-pilot"
    assert manifest["osfRegistrationUrl"] is None


@pytest.mark.integration
@pytest.mark.slow
def test_cli_validate_artifacts_after_run(tmp_path: Path) -> None:
    """validate-artifacts must pass for artifacts just written by the pilot."""
    runner = CliRunner()
    cfg = tmp_path / "pilot1.yaml"
    cfg.write_text(
        "run_label: pilot\nstatus: infrastructure-pilot\n"
        "seeds: [11]\ndataset_mode: synthetic\n"
        f"output_dir: {tmp_path / 'art'}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "art"

    run_result = runner.invoke(
        cli,
        ["pilot", "synthetic", "--config", str(cfg), "--out", str(out_dir)],
    )
    assert run_result.exit_code == 0

    val_result = runner.invoke(cli, ["validate-artifacts", "--dir", str(out_dir)])
    # validate-artifacts checks SHA256 hashes; all files should be consistent
    assert "MISMATCH" not in val_result.output
