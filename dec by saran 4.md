# `DR-003A-STAGE3-REMEDIATION` — Stage 3 Review Decision

**Decision authority:** Saranraj U  

**Stage 3 status:** **Provisionally accepted, exit gate not yet satisfied**  

**Stage 4 authorization:** **Withheld pending remediation**  

**Confirmatory experiments:** Not approved

The implementation appears substantially complete, but the submitted report does not satisfy several mandatory conditions from `DR-003-PHASE1-STAGE3`.

## Blocking issues

### 1. Stage 3 is not committed

The report says:

> Changes are staged/ready for commit.
> 

A staged working tree is not a frozen research artifact. Stage 3 requires:

- A complete 40-character commit hash
- A clean working tree
- The commit recorded in manifests and reports

### 2. Ubuntu CI evidence is missing

`DR-003` required testing on:

- Windows with Python 3.12
- Ubuntu CI with Python 3.12

Only Windows results were reported.

### 3. TreeSHAP additivity evidence is insufficient

The reported maximum residual of `< 0.05` is too loose for an additivity test and was reported only for Random Forest.

If expected value and attributions are expressed in the same output space, additivity should ordinarily hold near numerical precision. A residual approaching `0.05` may indicate:

- Probability versus raw-margin mismatch
- Incorrect binary-class extraction
- Wrong expected value
- Inconsistent background data
- Incompatible perturbation mode
- Incorrect output transformation

Both Random Forest and XGBoost require explicit results.

### 4. KernelSHAP and LIME validation is qualitative

“Directional agreement for `x1`” does not satisfy the requested analytic-reference comparison. Stage 3 requires quantitative agreement across all known signal features.

### 5. Stage sequencing is incorrect

The report identifies Stage 4 as:

> Real Dataset Loader Integrations
> 

The approved plan defines:

- **Stage 4:** Core metrics
- **Stage 5:** Stressors
- **Stage 6:** Dataset adapters

Do not begin real-dataset adapter work as Stage 4.

---

# Required remediation

Proceed with these corrections without requesting approval between ordinary steps.

## A. Freeze Stage 3

1. Commit all Stage 3 changes.
2. Run:

```bash
git rev-parse HEAD
git status --porcelain
uv lock --check
uv sync --locked --extra dev
```

1. Required result:

```
git status --porcelain
```

must return no output.

1. Record the complete commit hash in:
- Exit-gate report
- `run-manifest.json`
- `benchmark.json`
- `model-card.md`
- Environment record
- Task tracker

## B. Complete cross-platform CI

Run the full suite on Ubuntu with Python 3.12:

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run mypy src
uv run pytest -v --tb=short
```

Report:

- Operating-system image
- Python version
- Lockfile hash
- Test count
- Passed, failed, skipped and deselected counts
- Scientific-value differences between Windows and Ubuntu
- Declared comparison tolerances

Do not claim bitwise cross-platform reproducibility.

## C. Correct TreeSHAP additivity validation

Test both:

- Random Forest + TreeSHAP
- XGBoost + TreeSHAP

For every test sample, verify:

```
expected_value + sum(attributions) ≈ explained model output
```

The expected value, attributions and prediction must use the same output space.

### Required tolerance

Target:

```
maximum absolute residual ≤ 1e-6
```

An implementation-specific tolerance up to `1e-5` is acceptable if justified by floating-point behavior.

Do not increase the tolerance to make the current implementation pass. If residuals remain near `0.05`, treat that as an output-space or adapter defect.

Report separately:

| Model | Output space | Mean residual | P95 residual | Maximum residual |
| --- | --- | --- | --- | --- |
| --- | --- | ---: | ---: | ---: |
| Random Forest | Documented value |  |  |  |
| XGBoost | Documented value |  |  |  |

Also record:

- `model_output`
- Feature-perturbation mode
- Background-data hash
- Positive-class extraction procedure
- SHAP version

## D. Quantify KernelSHAP reference agreement

Use an independent linear synthetic model with raw-margin prediction:

```
f(x) = intercept + Σ w_j x_j
```

For independent features and a matching background expectation, the analytic attribution is:

```
φ_j = w_j × (x_j − E[X_j])
```

Compare KernelSHAP against this analytic reference.

Report:

- Mean absolute attribution error
- Maximum absolute attribution error
- Attribution-vector cosine similarity
- Spearman rank correlation
- Sign agreement across nonzero signal features
- Top-k feature agreement
- Results across multiple seeds and samples

Recommended Stage 3 gate:

```
mean cosine similarity ≥ 0.99
mean Spearman correlation ≥ 0.95
nonzero-feature sign agreement ≥ 0.95
```

If the gate fails, report the reason rather than changing the reference definition after seeing results.

## E. Quantify LIME reference agreement

For the same solvable linear fixture, report:

- Coefficient or attribution cosine similarity
- Spearman rank correlation
- Sign agreement across all nonzero signal features
- Top-k feature recall
- Variation across repeated LIME seeds

Recommended infrastructure gate:

```
mean cosine similarity ≥ 0.95
mean nonzero-feature sign agreement ≥ 0.90
top-k signal recall ≥ 0.90
```

Record:

- Kernel width
- Discretization
- Sampling budget
- Feature scaling
- Random seed
- Sparse-to-dense feature mapping

## F. Validate the complete compatibility matrix

Report test results for all eight supported combinations:

| Model | TreeSHAP | KernelSHAP | LIME |
| --- | --- | --- | --- |
| --- | ---: | ---: | ---: |
| Scikit-learn Logistic Regression | Not supported | Required | Required |
| Random Forest | Required | Required | Required |
| XGBoost | Required | Required | Required |

Also retain the custom logistic model with exact attribution and randomized control as a separate reference track.

## G. Audit static-analysis suppressions

List every new:

- Ruff per-file ignore
- `# noqa`
- `# type: ignore`
- Mypy override
- Third-party import suppression

For each suppression, record:

- File
- Rule
- Reason
- Whether it hides first-party code
- Removal plan

Suppressions may cover missing third-party stubs but must not hide type errors in ExplainCheck’s own contracts or scientific logic.

## H. Add Stage 3 validation artifact

Create:

```
artifacts/pilot/stage3-explainer-integration-v1/
└── integration-validation.json
```

It must include:

- Full commit hash
- Lockfile hash
- Windows and Ubuntu environments
- Compatibility matrix
- Additivity results
- KernelSHAP analytic-reference results
- LIME analytic-reference results
- Determinism results
- Failure-mode results
- Background hashes
- Output spaces
- Test commands and outcomes

Add it to `SHA256SUMS.txt` and rerun artifact validation.

---

# Correct next-stage boundary

After remediation, the next stage is:

## Stage 4 — Core Metrics

Authorized scope after Stage 3 approval will be:

- Cosine stability
- Spearman stability
- Top-k Jaccard completion
- k90 sparsity
- Runtime and memory measurement
- Cross-check against Quantus or OpenXAI
- Hand fixtures, property tests and golden tests

Do **not** begin:

- Full real-dataset adapters
- Correlation/missingness stressors
- Subgroup analyses
- Confirmatory benchmark runs

---

## Required corrected exit report

Return:

1. Full Stage 3 commit hash
2. Clean working-tree output
3. Windows and Ubuntu results
4. Full compatibility matrix
5. TreeSHAP additivity table for RF and XGBoost
6. Quantitative KernelSHAP reference results
7. Quantitative LIME reference results
8. Static-analysis suppression audit
9. Updated artifact validation
10. Failures, unresolved issues and deviations

> Complete this remediation and resubmit the Stage 3 exit-gate report. Do not proceed to Stage 4 until explicit approval is issued.
>