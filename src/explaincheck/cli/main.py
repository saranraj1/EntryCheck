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
@click.option(
    "--config", default="configs/pilot.yaml", type=click.Path(exists=True), help="YAML config path."
)
@click.option(
    "--out",
    default="artifacts/pilot/stage2-synthetic-v1",
    type=click.Path(),
    help="Output directory.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Validate config without running.")
def pilot_synthetic(config: str, out: str | None, dry_run: bool) -> None:
    """
    Run the synthetic linear-logit pilot (Stage 2).

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

    click.echo(
        f"ExplainCheck {__version__}  |  Study: {__study_id__}  |  Protocol: {__protocol_version__}"
    )
    click.echo(f"Config hash: {cfg_hash}")
    click.echo(f"Run label:   {cfg.run_label}")
    click.echo(f"Status:      {cfg.status}")
    click.echo(f"Seeds:       {cfg.seeds}")
    click.echo(f"Output dir:  {out}")

    if dry_run:
        click.echo("[DRY RUN] Config valid. No experiment executed.")
        return

    from pathlib import Path

    from explaincheck.pilot.runner import (
        check_reproduction,
        manual_validation,
        run_pilot,
        write_outputs,
    )

    click.echo("\n[1/4] Running manual validation fixture...")
    validation = manual_validation()
    click.echo(
        f"  Fidelity AOPC@2 = {validation['fidelity_aopc_at_2_computed']:.12f}  (expected 2.25) - {validation['status'].upper()}"
    )
    click.echo(
        f"  Stability Jaccard@2 = {validation['stability_jaccard_at_2_computed']:.12f}  (expected 1.0) - {validation['status'].upper()}"
    )

    seeds = cfg.seeds if cfg.seeds else [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

    click.echo(f"\n[2/4] Running pilot across {len(seeds)} seeds...")
    results, models, failures, timing = run_pilot(seeds=seeds)

    click.echo("\n[3/4] Checking Phase 0 reproduction tolerances...")
    reproduction = check_reproduction(results, models)
    for k, v in reproduction.items():
        if k.startswith("_"):
            continue
        status = "PASS" if v.get("passed") else "FAIL"
        click.echo(
            f"  [{status}]  {k}: got={v['got']}  ref={v['reference']}  tol=+-{v['tolerance']}"
        )

    if not reproduction.get("_all_passed"):
        click.echo(
            "\n[WARNING] Some reproduction checks FAILED. Inspect variance report in benchmark.json."
        )

    click.echo(f"\n[4/4] Writing outputs to {out}...")
    out_dir = Path(out) if out is not None else Path("artifacts/pilot/default")
    run_id = write_outputs(
        results, models, failures, validation, timing, reproduction, out_dir, config_hash=cfg_hash
    )

    click.echo(f"\n[DONE] Run ID: {run_id}")
    click.echo(f"       Elapsed: {timing['elapsed_seconds']:.1f}s")
    click.echo(f"       Artifacts: {out_dir.resolve()}")
    click.echo(f"       ROC AUC: {models.roc_auc.mean():.3f} ± {models.roc_auc.std():.3f}")
    click.echo("\n[NOTE] Label: infrastructure-pilot. Do not interpret as confirmatory evidence.")


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


@cli.command("validate-stage3")
@click.option(
    "--seeds",
    default="0,1,2,3,4",
    help="Comma-separated seed list for multi-seed validation (default: 0,1,2,3,4).",
)
@click.option(
    "--output-dir",
    default="artifacts/pilot/stage3-finalization-v1",
    type=click.Path(),
    help="Output directory for finalization artifacts.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    hidden=True,  # dev-only; must NOT be used for frozen confirmatory runs
    help="[DEV ONLY] Allow overwriting existing artifact directory.",
)
def validate_stage3(seeds: str, output_dir: str, overwrite: bool) -> None:
    """
    Run Stage 3 finalization validation (DR-006A §7).

    Executes multi-seed KernelSHAP and LIME validation and the compatibility
    determinism matrix, then writes canonical artifacts to --output-dir.

    Seeds: specified as comma-separated integers (default: 0,1,2,3,4).

    This command refuses to overwrite a non-empty output directory unless
    --overwrite is explicitly passed. That flag must never be used for
    frozen confirmatory runs.
    """
    import subprocess

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Guard: refuse to overwrite non-empty frozen artifact directory
    existing = list(out.iterdir())
    if existing and not overwrite:
        click.echo(
            f"[ERROR] Output directory '{out}' is non-empty "
            f"({len(existing)} file(s) found).\n"
            "This command will not overwrite frozen artifacts.\n"
            "Move or rename existing files, or use --overwrite [DEV ONLY].",
            err=True,
        )
        sys.exit(1)

    click.echo(f"validate-stage3: seeds={seed_list}, output-dir={out}")
    click.echo()

    scripts = [
        ("KernelSHAP multi-seed", "scripts/validation/stage3_kernelshap_multiseed.py"),
        ("LIME multi-seed", "scripts/validation/stage3_lime_multiseed.py"),
        ("Determinism matrix", "scripts/validation/stage3_determinism_matrix.py"),
    ]

    results: dict[str, bool] = {}
    seeds_arg = [str(s) for s in seed_list]

    for label, script_path in scripts:
        click.echo(f"--- {label} ---")
        cmd = [
            sys.executable,
            script_path,
            "--seeds",
            *seeds_arg,
            "--output-dir",
            str(out),
        ]
        ret = subprocess.run(cmd, capture_output=False)
        # exit 2 = gates failed (script ran correctly); exit 1 = error
        results[label] = ret.returncode in (0, 2)
        click.echo()

    click.echo("=== validate-stage3 summary ===")
    all_ran = all(results.values())
    for label, ran in results.items():
        status = "COMPLETE" if ran else "ERROR"
        click.echo(f"  {label}: {status}")
    click.echo()

    if all_ran:
        click.echo("All validation scripts completed. Review JSON artifacts for gate results.")
    else:
        click.echo("[WARNING] One or more scripts errored -- check output above.", err=True)
        sys.exit(1)


@cli.command("validate-stage3-artifacts")
@click.option(
    "--dir",
    "artifact_dir",
    default="artifacts/pilot/stage3-finalization-v1",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the stage3-finalization-v1 artifact directory.",
    show_default=True,
)
def validate_stage3_artifacts(artifact_dir: Path) -> None:
    """Validate stage3-finalization-v1 artifact checksums and JSON schema.

    Loads artifact-checksums.json (the trusted manifest), recomputes SHA-256
    for every listed file, compares expected vs observed hashes, validates
    required JSON fields, and detects missing or unexpected files.

    Exits nonzero if any check fails. The manifest itself is excluded from
    the checksum computation to avoid a self-referential hash problem.
    """
    import hashlib

    MANIFEST_NAME = "artifact-checksums.json"
    manifest_path = artifact_dir / MANIFEST_NAME

    if not manifest_path.exists():
        click.echo(f"[ERROR] Trusted manifest not found: {manifest_path}", err=True)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict] = manifest.get("files", {})
    required_fields: dict[str, list[str]] = manifest.get("required_json_fields", {})

    failures: list[str] = []
    checked = 0

    # 1. Check all expected files exist and hash matches
    TEXT_EXTENSIONS = {".json", ".csv", ".txt", ".md", ".py", ".toml", ".yml", ".yaml"}

    def _hash_file(path: Path) -> str:
        """SHA-256 of file content with CRLF→LF normalisation for text files."""
        raw = path.read_bytes()
        if path.suffix.lower() in TEXT_EXTENSIONS:
            raw = raw.replace(b"\r\n", b"\n")
        return hashlib.sha256(raw).hexdigest()

    for fname, meta in expected.items():
        fpath = artifact_dir / fname
        if not fpath.exists():
            failures.append(f"MISSING file: {fname}")
            continue
        observed = _hash_file(fpath)
        expected_sha = meta.get("sha256", "")
        if observed != expected_sha:
            failures.append(
                f"HASH MISMATCH: {fname}\n"
                f"  expected: {expected_sha}\n"
                f"  observed: {observed}"
            )
        else:
            click.echo(f"  [OK] {fname} (sha256: {observed[:16]}...)")
        checked += 1

    # 2. Detect unexpected files (not in manifest, not the manifest itself)
    actual_files = {f.name for f in artifact_dir.iterdir() if f.is_file()}
    expected_files = set(expected.keys()) | {MANIFEST_NAME}
    unexpected = actual_files - expected_files
    for fname in sorted(unexpected):
        failures.append(f"UNEXPECTED file (not in manifest): {fname}")

    # 3. JSON schema validation for required fields
    for fname, fields in required_fields.items():
        fpath = artifact_dir / fname
        if not fpath.exists():
            continue  # already flagged as missing above
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failures.append(f"INVALID JSON: {fname} — {e}")
            continue
        for field in fields:
            if field not in data:
                failures.append(f"MISSING JSON FIELD: {fname} missing '{field}'")

    # --- Summary ---
    click.echo()
    click.echo("=== validate-stage3-artifacts ===")
    click.echo(f"  Directory:  {artifact_dir}")
    click.echo(f"  Manifest:   {MANIFEST_NAME}")
    click.echo(f"  Files checked: {checked}")
    click.echo(f"  Failures:   {len(failures)}")

    if failures:
        click.echo("[FAIL]", err=True)
        for f in failures:
            click.echo(f"  {f}", err=True)
        sys.exit(1)
    else:
        click.echo("[PASS] All checksums match. All required JSON fields present.")


@cli.command("validate-stage4-infrastructure-artifacts")
@click.option(
    "--dir",
    "artifact_dir",
    default="artifacts/pilot/stage4-infrastructure-v1",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the stage4-infrastructure-v1 artifact directory.",
    show_default=True,
)
def validate_stage4_infrastructure_artifacts(artifact_dir: Path) -> None:
    """Validate stage4-infrastructure-v1 artifact checksums and JSON schema.

    Loads artifact-checksums.json (the trusted manifest), recomputes SHA-256
    for every listed file, compares expected vs observed hashes, validates
    required JSON fields, and detects missing or unexpected files.

    Exits nonzero if any check fails. The manifest itself is excluded from
    the checksum computation to avoid a self-referential hash problem.

    Does not modify Stage 3 artifacts.
    """
    import hashlib

    MANIFEST_NAME = "artifact-checksums.json"
    manifest_path = artifact_dir / MANIFEST_NAME

    if not manifest_path.exists():
        click.echo(f"[ERROR] Trusted manifest not found: {manifest_path}", err=True)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict] = manifest.get("files", {})
    required_fields: dict[str, list[str]] = manifest.get("required_json_fields", {})

    failures: list[str] = []
    checked = 0

    TEXT_EXTENSIONS = {".json", ".csv", ".txt", ".md", ".py", ".toml", ".yml", ".yaml"}

    def _hash_file(path: Path) -> str:
        """SHA-256 of file content with CRLF→LF normalisation for text files."""
        raw = path.read_bytes()
        if path.suffix.lower() in TEXT_EXTENSIONS:
            raw = raw.replace(b"\r\n", b"\n")
        return hashlib.sha256(raw).hexdigest()

    # 1. Check all expected files exist and hash matches
    for fname, meta in expected.items():
        fpath = artifact_dir / fname
        if not fpath.exists():
            failures.append(f"MISSING file: {fname}")
            continue
        observed = _hash_file(fpath)
        expected_sha = meta.get("sha256", "")
        if observed != expected_sha:
            failures.append(
                f"HASH MISMATCH: {fname}\n"
                f"  expected: {expected_sha}\n"
                f"  observed: {observed}"
            )
        else:
            click.echo(f"  [OK] {fname} (sha256: {observed[:16]}...)")
        checked += 1

    # 2. Detect unexpected files (not in manifest, not the manifest itself)
    actual_files = {f.name for f in artifact_dir.iterdir() if f.is_file()}
    expected_files = set(expected.keys()) | {MANIFEST_NAME}
    unexpected = actual_files - expected_files
    for fname in sorted(unexpected):
        failures.append(f"UNEXPECTED file (not in manifest): {fname}")

    # 3. JSON schema validation for required fields
    for fname, fields in required_fields.items():
        fpath = artifact_dir / fname
        if not fpath.exists():
            continue  # already flagged as missing above
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            failures.append(f"INVALID JSON: {fname} — {e}")
            continue
        for field in fields:
            if field not in data:
                failures.append(f"MISSING JSON FIELD: {fname} missing '{field}'")

    # --- Summary ---
    click.echo()
    click.echo("=== validate-stage4-infrastructure-artifacts ===")
    click.echo(f"  Directory:  {artifact_dir}")
    click.echo(f"  Manifest:   {MANIFEST_NAME}")
    click.echo(f"  Files checked: {checked}")
    click.echo(f"  Failures:   {len(failures)}")

    if failures:
        click.echo("[FAIL]", err=True)
        for f in failures:
            click.echo(f"  {f}", err=True)
        sys.exit(1)
    else:
        click.echo("[PASS] All checksums match. All required JSON fields present.")


@cli.command("env-snapshot")
@click.option(
    "--out", default=None, type=click.Path(), help="Write JSON to file instead of stdout."
)
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
