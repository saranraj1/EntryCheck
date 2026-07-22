"""
Generate artifact-checksums.json for artifacts/pilot/stage4-infrastructure-v1/.

This script computes SHA-256 for every file in the stage4 artifact directory
(excluding artifact-checksums.json itself) and writes a manifest that the
`validate-stage4-infrastructure-artifacts` CLI command can verify.

Usage:
    uv run python scripts/generate_stage4_checksums.py

The manifest is NOT self-referential: it excludes its own filename from
the hashes it contains, matching the behaviour of the Stage 3 validator.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def hash_file(path: Path) -> str:
    """SHA-256 of file content with CRLF->LF normalisation for text files."""
    TEXT_EXTENSIONS = {".json", ".csv", ".txt", ".md", ".py", ".toml", ".yml", ".yaml"}
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_EXTENSIONS:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    root = Path("artifacts/pilot/stage4-infrastructure-v1")
    if not root.is_dir():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    MANIFEST_NAME = "artifact-checksums.json"
    REQUIRED_FILES = [
        "README.md",
        "run-manifest.json",
        "infrastructure-validation.json",
        "test-inventory.json",
        "suppression-audit.json",
    ]

    REQUIRED_JSON_FIELDS = {
        "run-manifest.json": [
            "schema_version",
            "phase",
            "decision_record",
            "authorized_snapshot",
            "source_commit",
            "status",
            "confirmatory_experiments",
        ],
        "infrastructure-validation.json": [
            "schema_version",
            "platform",
            "python_version",
            "source_commit",
            "results",
            "ubuntu_validation",
            "suppression_summary",
        ],
        "test-inventory.json": [
            "schema_version",
            "source_commit",
            "test_count",
            "test_id_sha256",
            "ubuntu_validation",
        ],
        "suppression-audit.json": [
            "schema_version",
            "source_commit",
            "audit_scope",
            "quarantined_suppressions_removed",
            "remaining_suppressions",
            "verdict",
        ],
    }

    files: dict[str, dict[str, str]] = {}
    missing: list[str] = []

    for fname in REQUIRED_FILES:
        fpath = root / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            sha = hash_file(fpath)
            files[fname] = {"sha256": sha}
            print(f"  {fname}: {sha}")

    if missing:
        print(f"\nERROR: missing required files: {missing}", file=sys.stderr)
        sys.exit(1)

    # Check for unexpected files
    actual = {f.name for f in root.iterdir() if f.is_file()}
    expected_set = set(REQUIRED_FILES) | {MANIFEST_NAME}
    unexpected = actual - expected_set
    if unexpected:
        print(f"\nWARNING: unexpected files not added to manifest: {sorted(unexpected)}")

    manifest = {
        "schema_version": "1.0",
        "decision_record": "DR-008A",
        "artifact_directory": str(root),
        "files": files,
        "required_json_fields": REQUIRED_JSON_FIELDS,
        "_note": "artifact-checksums.json is excluded from its own hash list (self-referential hash problem).",
    }

    out_path = root / MANIFEST_NAME
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Files hashed: {len(files)}")


if __name__ == "__main__":
    main()
