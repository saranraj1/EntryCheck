"""Collect test IDs and compute a canonical hash for cross-platform comparison.

The hash is always the LAST line printed to stdout so CI can extract it with:
    ACTUAL=$(uv run python scripts/compute_test_id_hash.py | tail -1)

This is robust to pytest collection warnings polluting earlier stdout lines.
"""
from __future__ import annotations

import hashlib
import sys

import pytest


class IDCollector:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        for item in session.items:
            # Normalise to forward-slash paths
            self.ids.append(item.nodeid.replace("\\", "/"))


def main() -> None:
    collector = IDCollector()
    pytest.main(
        ["tests/", "--collect-only", "-q", "--no-header", "--tb=no"],
        plugins=[collector],
    )
    ids = sorted(collector.ids)
    print(f"Count: {len(ids)}", file=sys.stderr)
    payload = "\n".join(ids) + "\n"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"test-ID sha256: {h}", file=sys.stderr)
    # Write ONLY the bare hash to stdout (LAST line — CI extracts with tail -1)
    print(h, flush=True)


if __name__ == "__main__":
    main()
