"""
ExplainCheck provenance — hashing, environment snapshotting, and run ID generation.

Every hash embedded in a run manifest must be reproducible from the same inputs.
This module provides the canonical hashing functions for the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's binary content."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_string(s: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hash_bytes(s.encode("utf-8"))


def hash_json(obj: Any) -> str:
    """
    SHA-256 of a JSON-serialisable object.
    Keys are sorted for determinism.
    """
    serialised = json.dumps(obj, sort_keys=True, default=str)
    return hash_string(serialised)


def hash_array(arr: np.ndarray) -> str:
    """SHA-256 of a NumPy array's raw bytes (includes dtype and shape)."""
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def hash_directory(path: str | Path) -> dict[str, str]:
    """
    SHA-256 hashes for every file under a directory (recursive, sorted).
    Returns a dict mapping relative path string → sha256 hex.
    """
    root = Path(path)
    result: dict[str, str] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            key = str(f.relative_to(root))
            result[key] = hash_file(f)
    return result


# ---------------------------------------------------------------------------
# Run ID and timestamps
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Generate a unique run ID: 'ec-<8-char-uuid>'."""
    return f"ec-{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def snapshot_environment() -> dict[str, str]:
    """
    Capture all library versions relevant to the study.
    Returns a dict suitable for embedding in a RunManifest.
    """
    import importlib.metadata as meta

    def ver(pkg: str) -> str:
        try:
            return meta.version(pkg)
        except meta.PackageNotFoundError:
            return "not-installed"

    return {
        "python_version": sys.version,
        "numpy_version": ver("numpy"),
        "pandas_version": ver("pandas"),
        "sklearn_version": ver("scikit-learn"),
        "xgboost_version": ver("xgboost"),
        "shap_version": ver("shap"),
        "lime_version": ver("lime"),
        "matplotlib_version": ver("matplotlib"),
        "pydantic_version": ver("pydantic"),
        "platform": platform.platform(),
        "cpu_count": str(platform.machine()),
    }


# ---------------------------------------------------------------------------
# Artifact registration
# ---------------------------------------------------------------------------


def register_artifacts(directory: str | Path) -> dict[str, dict[str, Any]]:
    """
    Scan a directory and return a dict of {relative_path: {sha256, bytes}}
    for embedding in a run manifest.
    """
    root = Path(directory)
    entries: dict[str, dict[str, Any]] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.name != "run-manifest.json":
            entries[str(f.relative_to(root))] = {
                "sha256": hash_file(f),
                "bytes": f.stat().st_size,
            }
    return entries
