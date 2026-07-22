# Stage 4 Infrastructure Pilot — Artifact Directory

**Decision record:** DR-008 (2026-07-22, Saranraj U)  
**Phase:** Phase 1 — Infrastructure Development (pre-confirmatory)  
**Authorized snapshot:** `cdc6a0070ac70feaa4fa19ba33cd131d69beaadd`

---

## Contents

| File | Purpose |
|---|---|
| `README.md` | This file — directory scope and provenance |
| `run-manifest.json` | Phase, decision record, source commit, status |
| `infrastructure-validation.json` | Windows engineering validation evidence |
| `test-inventory.json` | Test-ID hash and count |
| `suppression-audit.json` | Classified suppression audit result |

---

## Append-only notice

This directory is **append-only**. Once a file is committed here, its content
must not be overwritten. New evidence is appended as new files with distinct names.

Stage 3 artifacts remain in `artifacts/pilot/stage3-finalization-v1/` and are untouched.

---

## What this pilot covers

- Migration of 4 quarantined Stage 4 metrics to the Option B+ typed context interface
- Removal of all 4 first-party `# type: ignore[override]` suppressions
- New `PairwiseStabilityContext`, `SparsityContext`, `RuntimeContext` classes
- Context-contract tests for immutability, type validation, rejection of invalid inputs
- Coverage amendment disclosure (DR-007 ratification)

## What this pilot does NOT cover

- Confirmatory experiments (NOT AUTHORIZED)
- Real-dataset benchmarks (NOT AUTHORIZED)
- Stage 4 scientific exit validation (requires separate plan and approval)
