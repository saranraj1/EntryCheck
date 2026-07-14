"""
ExplainCheck CLI — entry point for all experiment commands.

Usage:
    explaincheck --help
    explaincheck pilot synthetic --config configs/pilot.yaml --out artifacts/pilot/
    explaincheck smoke --config configs/smoke.yaml
    explaincheck validate-artifacts --dir artifacts/pilot/
    explaincheck env-snapshot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from explaincheck import __protocol_version__, __study_id__, __version__


@click.group()
@click.version_option(__version__, prog_name="explaincheck")
def cli() -> None:
    """ExplainCheck — tabular XAI benchmark (EC-TABULAR-001)."""


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------

@cli.group()
def pilot() -> None:
    """Pilot experiment commands."""


@pilot.command("synthetic")
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config path.")
@click.option("--out", default=None, type=click.Path(), help="Output directory override.")
@click.option("--dry-run", is_flag=True, default=False, help="Validate config without running.")
def pilot_synthetic(config: str, out: str | None, dry_run: bool) -> None:
    """
    Run the synthetic linear-logit pilot.

    Reproduces Phase 0 results within declared tolerance.
    Label: infrastructure-pilot (not confirmatory).
    """
    from explaincheck.config import load_config

    cfg, cfg_hash = load_config(config)

    if cfg.status not in {"pilot", "infrastructure-pilot"}:
        click.echo(
            f"[ERROR] Config status='{cfg.status}' is not a pilot label. "
            "Use a pilot or smoke config for non-confirmatory runs.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"ExplainCheck {__version__}  |  Study: {__study_id__}  |  Protocol: {__protocol_version__}")
    click.echo(f"Config hash: {cfg_hash}")
    click.echo(f"Run label:   {cfg.run_label}")
    click.echo(f"Status:      {cfg.status}")
    click.echo(f"Seeds:       {cfg.seeds}")

    if dry_run:
        click.echo("[DRY RUN] Config valid. No experiment executed.")
        return

    # Stage 2 implementation will import and run the pilot here.
    click.echo("[STUB] Synthetic pilot runner — Stage 2 implementation pending.")


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

@cli.command("smoke")
@click.option("--config", default="configs/smoke.yaml", type=click.Path(exists=True))
def smoke(config: str) -> None:
    """Run a fast infrastructure smoke test (< 60 seconds)."""
    from explaincheck.config import load_config

    cfg, cfg_hash = load_config(config)
    click.echo(f"[SMOKE] config={config} hash={cfg_hash} seeds={cfg.seeds}")
    click.echo("[STUB] Smoke runner — Stage 2 implementation pending.")


# ---------------------------------------------------------------------------
# validate-artifacts
# ---------------------------------------------------------------------------

@cli.command("validate-artifacts")
@click.option("--dir", "artifact_dir", required=True, type=click.Path(exists=True))
def validate_artifacts(artifact_dir: str) -> None:
    """Verify SHA-256 checksums for a frozen artifact directory."""
    from explaincheck.provenance import hash_file

    root = Path(artifact_dir)
    manifest_path = root / "run-manifest.json"
    if not manifest_path.exists():
        click.echo(f"[ERROR] run-manifest.json not found in {root}", err=True)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    n_ok, n_fail = 0, 0

    for rel_path, meta in files.items():
        full = root / rel_path
        if not full.exists():
            click.echo(f"  MISSING  {rel_path}")
            n_fail += 1
            continue
        actual = hash_file(full)
        expected = meta.get("sha256", "")
        if actual == expected:
            click.echo(f"  OK       {rel_path}")
            n_ok += 1
        else:
            click.echo(f"  MISMATCH {rel_path}  expected={expected[:12]}…  got={actual[:12]}…")
            n_fail += 1

    click.echo(f"\n{n_ok} OK, {n_fail} FAILED")
    if n_fail:
        sys.exit(1)


# ---------------------------------------------------------------------------
# env-snapshot
# ---------------------------------------------------------------------------

@cli.command("env-snapshot")
@click.option("--out", default=None, type=click.Path(), help="Write JSON to file instead of stdout.")
def env_snapshot(out: str | None) -> None:
    """Print a JSON snapshot of the current environment for provenance."""
    from explaincheck.provenance import snapshot_environment

    env = snapshot_environment()
    payload = json.dumps(env, indent=2)
    if out:
        Path(out).write_text(payload, encoding="utf-8")
        click.echo(f"Environment snapshot written to {out}")
    else:
        click.echo(payload)


if __name__ == "__main__":
    cli()
