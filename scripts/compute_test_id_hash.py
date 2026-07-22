"""Collect test IDs and compute a canonical hash for cross-platform comparison."""
from __future__ import annotations

import hashlib
import os
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
    # Redirect pytest stdout to devnull: prevents shap DeprecationWarnings on
    # Ubuntu from polluting the stdout hash output used by CI comparison.
    with open(os.devnull, "w") as devnull:  # noqa: PTH123
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            pytest.main(
                ["tests/", "--collect-only", "-q", "--no-header", "--tb=no"],
                plugins=[collector],
            )
        finally:
            sys.stdout = old_stdout
    ids = sorted(collector.ids)
    print(f"Count: {len(ids)}", file=sys.stderr)
    payload = "\n".join(ids) + "\n"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"test-ID sha256: {h}", file=sys.stderr)
    # Write ONLY the bare hash to stdout (used by CI hash comparison)
    print(h)


if __name__ == "__main__":
    main()
