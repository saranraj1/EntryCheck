# `DR-003B-STAGE3-REMEDIATION-EXECUTION` — Plan Approval

**Decision:** Remediation plan approved with the corrections below.  

**Stage 4:** Still blocked.  

**Confirmatory experiments:** Not approved.

Proceed through all remediation steps without requesting approval between ordinary implementation actions.

## Required corrections to the plan

### 1. Avoid commit-hash self-reference

Do not attempt to embed a commit hash inside files that belong to that same commit. Committing the modified files would produce a new hash.

Use a two-commit sequence:

1. **Source commit**
    - Commit all Stage 3 source code, configuration and tests.
    - Record this as `sourceCommit`.
2. **Artifact commit**
    - Generate validation artifacts using `sourceCommit`.
    - Commit the artifacts, report and checksums.
    - Record this separately as `artifactCommit` in the exit report.

The generated manifest should contain:

```json
{
  "sourceCommit": "<full source-code commit>",
  "artifactGeneratedFromCleanSource": true
}
```

It does not need to contain its own artifact-commit hash.

The final exit report must provide:

- Stage 2 reference commit
- Provenance-correction commit
- Stage 3 source commit
- Stage 3 artifact commit
- Final clean working-tree status

### 2. Ubuntu CI is mandatory

A configured workflow is not evidence that the workflow passed.

A local Docker run may be used as an additional diagnostic, but it does **not** replace the required GitHub Actions result.

Required approach:

1. Confirm that `tests.yml` supports:

```yaml
workflow_dispatch:
```

1. Configure:

```yaml
runs-on: ubuntu-latest
```

with Python 3.12 and the locked environment.

1. Push the Stage 3 source branch/commit to GitHub.
2. Trigger the workflow by push or manual dispatch.
3. Record:
    - Workflow URL
    - Run ID
    - Commit tested
    - Ubuntu image
    - Python version
    - Lockfile hash
    - Test, Ruff and Mypy results
    - Final workflow conclusion

If GitHub Actions cannot be triggered, report Ubuntu CI as **unresolved and blocking**. Do not describe Stage 3 as complete.

### 3. Do not predeclare the checksum count

Do not assume the result will be `19/19`.

Adding `integration-validation.json` and modifying manifests may change the number of artifacts. Validate every expected artifact and report the computed total:

```
N/N artifacts valid, 0 failed
```

`SHA256SUMS.txt` should not contain a checksum for itself.

### 4. TreeSHAP output-space validation

For probability-space additivity, use a consistent configuration such as:

- Training-only background data
- Interventional feature perturbation
- `model_output="probability"`
- Positive-class expected value
- Positive-class attribution tensor
- `predict_proba(X)[:, 1]` as the comparison output

Handle SHAP’s possible binary-output shapes explicitly. Do not infer the positive-class axis solely from array dimensions without testing it.

If probability-space additivity remains problematic, test raw-margin additivity separately. Never compare raw-margin attributions with probability predictions.

Report probability-space and raw-space results separately if both are supported.

### 5. KernelSHAP analytic test

The closed-form reference

```
φ_j = w_j × (x_j − E[X_j])
```

is valid for the linear **raw-margin** function.

Therefore, the analytic test must explain:

```
f(x) = intercept + X @ w
```

not `predict_proba`, unless a separate probability-space reference is mathematically derived.

Use:

- Identity link
- One-row mean background or an analytically equivalent background distribution
- Fixed sampling budget
- Multiple samples and seeds
- Full-vector comparison, not only dominant features

### 6. LIME analytic test

For the analytic linear-reference test:

- Set `discretize_continuous=False`
- Record whether features are standardized
- Convert LIME coefficients back into the common original-feature representation when scaling is used
- Compare all nonzero signal features
- Report variation across seeds
- Verify sparse-to-dense feature mapping

Do not compare discretized interval explanations directly with raw linear coefficients.

### 7. Compatibility evidence

Each compatibility-matrix cell must include:

- Test name
- Model version
- Explainer version
- Output space
- Target class
- Background hash where applicable
- Determinism status
- Final pass/fail result

“Not supported” must also be tested as a deliberate structured rejection.

### 8. Suppression audit requirement

No suppression may hide first-party scientific logic.

Particularly inspect:

```
# type: ignore[override]
disable_error_code
ruff per-file-ignores
```

Prefer correcting base-class signatures over suppressing override incompatibilities. If a suppression remains, provide a narrowly scoped reason and removal issue.

## Execution authorization

Proceed with:

1. Source commit and clean-tree verification
2. TreeSHAP diagnosis and correction
3. Quantitative KernelSHAP validation
4. Quantitative LIME validation
5. Complete compatibility testing
6. Static-analysis suppression audit
7. Integration-validation artifact generation
8. Artifact commit
9. GitHub Actions Ubuntu run
10. Corrected Stage 3 exit report

## Completion rule

Do not report Stage 3 as complete unless:

- Windows checks pass
- Ubuntu GitHub Actions checks pass
- TreeSHAP additivity passes for Random Forest and XGBoost
- KernelSHAP and LIME pass quantitative analytic-reference gates
- Compatibility evidence is complete
- Suppressions are audited
- Artifacts validate
- Both full commit hashes are reported
- Final working tree is clean

> **Proceed with remediation now. If Ubuntu CI cannot be run, complete all other work and report Ubuntu CI as the sole remaining blocker. Do not begin Stage 4.**
>