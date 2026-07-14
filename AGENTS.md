# ExplainCheck agent instructions

## Mandatory startup reads (in order)

1. `PROTOCOL_V1.0.md`
2. `PREREGISTRATION_READY.md`
3. `COMPARISON.md`
4. `DATASET_REVIEW.md`
5. `PUBLICATION_STRATEGY.md`
6. `dec by saran 1.md` (Decision Record DR-001-PHASE1)
7. `amendments/` directory — read all amendment files
8. Latest `artifacts/**/run-manifest.json`

## Current phase

Phase 1 — Infrastructure Development (pre-confirmatory).
Decision Record: `DR-001-PHASE1` (2026-07-14, Saranraj U).

## Core rules

The project is publication-first. Saranraj U is the accountable researcher and author. AI agents assist but never own scientific decisions or authorship.

Never change frozen research scope silently. Never overwrite raw or confirmatory artifacts. Never invent evidence. Keep local and global explanations separate, condition stability analysis on prediction preservation, preserve failures, and generate all manuscript numbers from frozen artifacts.

Before editing, report the current phase, research question, task classification, planned files, tests, outputs, and approval requirements.

## Frozen decisions (from DR-001-PHASE1)

- GBT implementation: **XGBoost 2.1.4** (frozen — do not upgrade)
- Python version: **3.12** (not 3.13)
- Package manager: **uv** with `pyproject.toml` + `uv.lock`
- Architecture: modular `src/explaincheck/` with Pydantic typed contracts
- H1 wording: "No single explainer dominates across all **seven** prespecified evaluation dimensions" (see `amendments/2026-07-14-pre-registration-clarification.md`)
- Confirmatory runs: **NOT APPROVED** — requires OSF registration + all exit gates + Saranraj U explicit approval

## What requires Saranraj U approval

- Any frozen scope change (datasets, models, explainers, metrics, seeds, stress levels, exclusion rules)
- Starting confirmatory runs
- External preregistration deposit
- Paper submission or public release with new scientific claims
- Changing any threshold after inspecting relevant confirmatory outcomes