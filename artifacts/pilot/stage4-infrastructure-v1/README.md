# Stage 4 Infrastructure Pilot — Artifact Directory

**Decision record:** DR-008A (2026-07-22, Saranraj U)  
**Phase:** Phase 1 — Infrastructure Development (pre-confirmatory)  
**Final source commit:** `14264350431ebec345646ac4c097e818512ce7b3`

---

## Contents

| File | Purpose |
|---|---|
| `README.md` | This file — directory scope and provenance |
| `run-manifest.json` | Phase, decision record, source commit, status |
| `infrastructure-validation.json` | Windows engineering validation evidence |
| `test-inventory.json` | Test-ID hash, test count, new test names |
| `suppression-audit.json` | Full classified suppression audit (DR-008A §5) |
| `artifact-checksums.json` | SHA-256 manifest for this directory |

---

## Append-only notice

This directory is **append-only**. Once a file is committed here, its content
must not be overwritten. New evidence is appended as new files with distinct names.

Stage 3 artifacts remain in `artifacts/pilot/stage3-finalization-v1/` and are untouched.

---

## What this pilot covers

- Migration of 4 quarantined Stage 4 metrics to the Option B+ typed context interface
- Removal of all 4 first-party `# type: ignore[override]` suppressions
- Removal of 3 newly-introduced dead `# noqa: ANN401` markers (DR-008A correction)
- New `PairwiseStabilityContext`, `SparsityContext`, `RuntimeContext` classes
- Context-contract tests for immutability, type validation, rejection of invalid inputs
- Coverage amendment disclosure with Saranraj U countersignature (DR-008A §7)
- `validate-stage4-infrastructure-artifacts` CLI command
- Cross-platform CI: Ubuntu test-ID hash compared to expected value automatically

## What this pilot does NOT cover

- Confirmatory experiments (NOT AUTHORIZED)
- Real-dataset benchmarks (NOT AUTHORIZED)
- Stage 4 scientific exit validation (requires separate plan and approval)
