
# `DR-003-PHASE1-STAGE3` — Stage 2 Acceptance and Stage 3 Authorization

**Decision date:** 2026-07-14  

**Decision authority:** Saranraj U  

**Study:** `EC-TABULAR-001`  

**Stage 2 status:** Approved based on the submitted exit-gate report  

**Stage 3 status:** Infrastructure integration approved  

**Confirmatory experiments:** Not approved

## 1. Stage 2 decision

**Decision: APPROVED, subject to provenance completion.**

The following evidence is accepted:

- 68/68 reported tests passed
- Both hand-computed fixtures passed exactly
- All six Phase 0 regression checks passed within tolerance
- 18/18 artifact checksums validated
- No pipeline failures remained unresolved
- Phase 0 artifacts were reportedly preserved
- Stage 2 artifacts were written to a separate pilot directory
- The run is correctly labelled `pilot-not-confirmatory`
- No confirmatory experiment was executed
- No scientific protocol deviation was reported

### Required provenance correction

Before freezing Stage 2:

1. Replace the abbreviated Stage 2 commit hash with the complete 40-character hash.
2. Record the full hash in:
    - Stage 2 exit-gate report
    - `run-manifest.json`
    - `benchmark.json`
    - `model-card.md`
    - Stage 2 task tracker
3. Run:

```bash
git rev-parse HEAD
git status --short
uv lock --check
uv sync --locked --extra dev
uv run explaincheck validate-artifacts artifacts/pilot/stage2-synthetic-v1
```

1. Confirm that the working tree is clean after committing the provenance correction.
2. Generate a new checksum only for modified metadata files; do not modify scientific result files.

The NumPy and scikit-learn version differences from Phase 0 are accepted because every declared reproduction result remains within its prespecified tolerance.

## 2. Stage 3 scope

Stage 3 is authorized as an **infrastructure-only explainer and model integration stage**.

Implement:

### Model adapters

- Scikit-learn Logistic Regression reference adapter
- Random Forest classifier adapter
- XGBoost classifier adapter

The custom gradient-descent logistic implementation must remain available as the Phase 0/Stage 2 reproduction reference.

### Explainer adapters

- TreeSHAP
- KernelSHAP
- LIME

### Dataset work allowed during Stage 3

Dataset-loader contracts and small infrastructure smoke fixtures may be implemented. Full dataset-adapter validation remains Stage 6 work.

Real UCI data may be downloaded only for:

- Schema inspection
- License and DOI verification
- Loader smoke tests
- Feature-lineage tests
- Small explicitly labelled infrastructure tests

Do not run the frozen real-dataset benchmark matrix, hypothesis tests, subgroup analyses, or confirmatory stress experiments.

## 3. Frozen integration decisions

### TreeSHAP

Use TreeSHAP only with compatible tree models:

- Random Forest
- XGBoost

Requirements:

- Explain the positive-class probability or a precisely documented raw-margin output.
- Do not mix output spaces across models.
- Record `model_output`, perturbation mode, background data, target class and SHAP version.
- Test SHAP additivity in the selected output space.
- Convert binary-class output consistently into the common `ExplanationBatch` contract.

### KernelSHAP

Use KernelSHAP as the common model-agnostic explainer.

Infrastructure default:

- Background data must come from the training partition only.
- Use a deterministic seeded sample of 50 training rows for Stage 3 tests.
- Use a small fixed `nsamples` budget in CI.
- Never use test rows as background data.
- Record background-row hashes and sampling parameters.
- Treat production/confirmatory sampling budgets as not yet approved.

### LIME

Requirements:

- Use `LimeTabularExplainer`.
- Pass the random seed explicitly.
- Fit explainer statistics from training data only.
- Return a dense attribution vector aligned with the transformed feature schema.
- Record kernel width, discretization, sampling budget, target class and feature mapping.
- Do not silently discard features omitted from the sparse LIME explanation; represent them as zero in the aligned dense vector.
- Keep categorical-feature handling explicit.

## 4. Common explanation contract

Every adapter must return a common typed object containing at least:

```
sample IDs
target class
prediction
prediction/output space
base value when available
ordered feature names
dense attribution matrix
explainer name and version
model hash
background-data hash
parameters
seed
generation runtime
status and warnings
```

Reject outputs when:

- Attribution width differs from feature-schema width
- Feature order cannot be verified
- NaN or infinite attributions are returned
- Target class is ambiguous
- Output space is unspecified
- Model or background hashes are absent
- Local and global explanation scopes are mixed

## 5. Model requirements

### Random Forest

- Pass `random_state`
- Use `n_jobs=1` for controlled runtime measurements
- Record number of estimators, depth, splitting criteria and class weighting
- Test probability and class-label consistency

### XGBoost

Use frozen `xgboost==2.1.4`.

Requirements:

- Pass `random_state` and all relevant seeds
- Use CPU execution
- Use `n_jobs=1` for controlled timing
- Record objective, evaluation metric, tree method and all hyperparameters
- Disable early stopping in basic deterministic adapter tests unless the validation procedure is explicitly frozen
- Test repeated fits for prediction and model-hash consistency

### Scikit-learn Logistic Regression

- Use a deterministic solver
- Record solver, regularization, class weighting, iteration limit and convergence status
- Treat convergence warnings as structured warnings, not console noise

## 6. Required Stage 3 tests

### Contract tests

For every compatible model–explainer combination:

- Output shape
- Feature ordering
- Target class
- Output space
- Base-value handling
- Dense attribution representation
- Serialization
- Hash and provenance presence

### Determinism tests

Run identical seeded calls twice and verify:

- Predictions match
- Attributions match exactly when deterministic
- Otherwise, values remain within a prespecified numerical tolerance
- Manifests and non-time provenance fields match

Do not require runtime timestamps or elapsed durations to be identical.

### Scientific sanity tests

- TreeSHAP additivity on Random Forest and XGBoost
- TreeSHAP recovery on a simple synthetic tree model
- KernelSHAP agreement with the exact-linear reference on a small linear fixture
- LIME directional agreement on an analytically solvable linear fixture
- Randomized negative control remains worse than the exact reference under the validated pilot metrics
- Changing a genuinely influential feature changes its explanation in the expected direction
- Noise features do not systematically outrank known signal features

### Failure tests

Test:

- Unsupported model for TreeSHAP
- Missing background data
- Test-data leakage into background data
- Wrong target class
- Feature-schema mismatch
- Missing feature lineage
- NaN and infinite values
- Degenerate constant model
- Single-class target
- Explainer timeout or sampling failure
- Model prediction failure

Failures must be written as structured records.

## 7. Cross-platform requirement

Stage 3 must test at least:

- Windows with Python 3.12
- Ubuntu CI with Python 3.12

The environment lock remains shared, but manifests must record platform-specific details.

Do not claim bitwise cross-platform identity. Compare scientific values using declared tolerances and report any platform-dependent behavior.

## 8. Artifact boundary

Write outputs to a new directory such as:

```
artifacts/pilot/stage3-explainer-integration-v1/
```

Do not modify:

```
ExplainCheck-Phase0-Research-Package/
artifacts/pilot/stage2-synthetic-v1/
```

Stage 3 artifacts must be labelled:

```json
{
  "status": "infrastructure-pilot",
  "confirmatory": false,
  "osfRegistrationUrl": null
}
```

## 9. Stage 3 quality commands

At minimum, run:

```bash
uv lock --check
uv sync --locked --extra dev
uv run ruff check .
uv run mypy src
uv run pytest -v --tb=short
uv run explaincheck validate-artifacts artifacts/pilot/stage3-explainer-integration-v1
```

Run integration and scientific tests on both Windows and Ubuntu CI.

## 10. Stage 3 exit gate

Stage 3 is complete only when:

- Stage 2 provenance contains the full commit hash
- Working tree and lockfile state are recorded
- Random Forest, XGBoost and reference Logistic Regression adapters pass
- TreeSHAP, KernelSHAP and LIME adapters pass their compatible model matrix
- Analytic-reference tests pass
- TreeSHAP additivity passes in the documented output space
- No training/test leakage occurs
- Feature lineage is preserved
- Failure cases produce structured records
- Windows and Ubuntu tests pass
- All artifacts validate and are checksummed
- No frozen real-dataset benchmark is executed
- No scientific protocol decision is changed
- No unresolved critical or high-severity issue remains

## 11. Required exit report

Return:

1. Full Stage 2 and Stage 3 commit hashes
2. Working-tree status
3. Lockfile and environment hashes
4. Model–explainer compatibility matrix
5. Files created and modified
6. Complete test and CI results
7. Analytic-reference comparison results
8. TreeSHAP additivity results
9. Artifact paths and checksums
10. Failures and unresolved issues
11. Protocol deviations
12. Required human decisions

## Final authorization

> Stage 2 is approved based on the submitted exit-gate report. Complete the full-hash provenance correction, then proceed with Stage 3 infrastructure integration without requesting approval between ordinary implementation steps. Do not begin confirmatory experiments, execute the frozen real-dataset benchmark matrix, or proceed to Stage 4 until the Stage 3 exit report is reviewed and explicitly approved.
>