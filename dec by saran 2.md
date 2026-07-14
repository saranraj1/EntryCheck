Stage 1 is **accepted based on the supplied report**, but not independently verified because I cannot access the local Windows repository.

Send this authorization to Antigravity:

---

# `DR-002-PHASE1-STAGE2` — Stage 2 Authorization

**Decision authority:** Saranraj U  

**Status:** Stage 2 infrastructure work approved  

**Confirmatory experiments:** Not approved

Proceed with **Stage 2 — Synthetic Scientific Baseline** without requesting approval between implementation steps. Stop only if:

- Frozen protocol content would change
- Phase 0 behavior cannot be reproduced
- A scientific definition is ambiguous
- Existing frozen artifacts would be overwritten
- A critical dependency or test failure cannot be resolved

## Before Stage 2

1. Verify whether `uv.lock` already exists. `uv sync` normally generates it.
2. Run:

```bash
uv lock --check
uv sync --locked --extra dev
git status
```

1. Commit the completed Stage 1 foundation and record the commit hash.
2. Update the environment snapshot and manifest with:
    - Python version
    - OS
    - XGBoost, SHAP, LIME, NumPy, pandas, scikit-learn and Pydantic versions
    - `uv.lock` hash
    - Git commit hash
3. Do not describe the environment as frozen until `uv.lock` exists and is committed.

## Stage 2 authorized work

Implement:

- Synthetic linear dataset generator
- Deterministic train/test splitting
- Logistic Regression model adapter
- Exact-linear attribution adapter
- Randomized within-instance negative control
- Deletion fidelity AOPC
- Prediction-conditioned Top-k Jaccard stability
- Pilot runner
- Benchmark JSON generation
- Tidy CSV generation
- Model-card generation
- Methods and Results snippet generation
- Run manifest and checksums
- CLI command for reproducing the synthetic pilot

## Scientific requirements

- Migrate the Phase 0 implementation; do not silently rewrite its scientific definitions.
- Preserve Phase 0 artifacts as immutable reference artifacts.
- Write Stage 2 outputs to a new directory.
- Keep the randomized method clearly labelled as a negative control.
- Do not interpret Stage 2 results as evidence about SHAP, LIME, XGBoost or real-world datasets.
- Record prediction-preservation rates.
- Record all seeds, parameters, timings and failures.
- Use the frozen ten seeds.

## Required validation

### Hand fixtures

These must pass exactly:

```
Fidelity AOPC@2 = 2.25
Stability Jaccard@2 = 1.0
```

### Phase 0 reproduction tolerances

At `n=200`, reproduce approximately:

| Result | Reference | Tolerance |
| --- | --- | --- |
| --- | ---: | ---: |
| Mean ROC AUC | 0.878 | ±0.005 |
| Coefficient cosine | 0.998 | ±0.002 |
| Exact fidelity | 1.764 | ±0.020 |
| Negative-control fidelity | 0.753 | ±0.030 |
| Exact stability | 0.957 | ±0.015 |
| Negative-control stability | 0.259 | ±0.020 |

If a result falls outside tolerance, do not adjust the tolerance or implementation to force a pass. Produce a variance report identifying:

- Dependency differences
- RNG differences
- Model-fitting differences
- Numerical precision
- Data-split differences
- Attribution or metric-definition differences

## Required tests

Add and execute:

- Unit tests
- Property-based tests
- Golden scientific tests
- Determinism tests
- Invalid-input tests
- Artifact-schema tests
- CLI integration tests
- Phase 0 regression tests

Run the complete suite, not only `tests/unit/`:

```bash
uv run pytest -v --tb=short
uv run ruff check .
uv run mypy src
uv run explaincheck pilot synthetic
uv run explaincheck validate-artifacts <stage-2-artifact-directory>
```

## Stage 2 output directory

Use a distinct path such as:

```
artifacts/pilot/stage2-synthetic-v1/
```

It must contain:

```
run-manifest.json
environment.json
benchmark.json
raw-results.parquet
tidy-results.csv
model-performance.csv
failures.csv
manual-metric-validation.json
model-card.md
SHA256SUMS.txt
tables/
figures/
paper-snippets/
```

## Stage 2 exit gate

Stage 2 is complete only when:

- All tests and quality checks pass
- `uv.lock` is committed
- The Git commit is recorded
- Hand fixtures pass exactly
- Phase 0 results reproduce within tolerance or a justified variance report exists
- Artifacts validate against their schemas
- All generated files have checksums
- Phase 0 artifacts remain unchanged
- No confirmatory experiment has been executed
- No protocol deviation occurred

At completion, return one consolidated report containing:

1. Commit hash
2. Environment and lockfile hash
3. Files created or modified
4. Complete test results
5. Reproduction comparison table
6. Artifact paths and checksums
7. Failures and unresolved issues
8. Protocol deviations
9. Required human review

**Proceed with Stage 2 now. Do not begin Stage 3 until the Stage 2 exit-gate report has been reviewed.**