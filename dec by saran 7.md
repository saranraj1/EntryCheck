[Certain] The only remaining weakness is that the Ubuntu hash is reported rather than directly reproduced in the pasted log; this is now a documentation-quality issue, not a defensible reason to block Stage 3 again.

# DR-007 — Stage 3 Exit Approval and Stage 4 Authorization

**Project:** EC-TABULAR-001 — ExplainCheck  

**Date:** 2026-07-22  

**Accountable researcher:** Saranraj U  

**Stage 3 decision:** ✅ **Approved and closed**  

**Stage 4 decision:** ✅ **Authorized to resume under restrictions**  

**Confirmatory experiments:** ❌ **Not authorized**

## 1. Final determination

[Certain] Stage 3 has satisfied its scientific, engineering, reproducibility and provenance exit requirements based on the submitted DR-006A through DR-006D evidence.

[Certain] The final accepted repository snapshot is:

```
cdc6a0070ac70feaa4fa19ba33cd131d69beaadd
```

[Certain] This snapshot passed the reported Windows and Ubuntu workflows with:

- 155 collected tests
- 155 passing tests
- Matching test inventory:
    
    `d9017aa1bd4073cfd7ac4280483663c7f4bd3c70e7b2198fbd5ea83408058b80`
    
- Ruff lint and format checks
- Mypy checks
- Coverage above 80%
- Seven-of-seven artifact validation
- Matching lockfile:
    
    `b617a49ba2d6e40d34e3a92c4078bf64c90862b4e21fdf5cb2612e9b1f0c63b1`
    

[Certain] The earlier 100-row LIME background entry is superseded. The accepted configuration is:

- KernelSHAP background: 50 rows
- LIME reference-validation background: 200 rows
- RF+LIME determinism background: 200 rows
- XGB+LIME determinism background: 200 rows

## 2. Accepted provenance map

| Role | Accepted commit |
| --- | --- |
| Scientific Stage 3 source | `e2e0b0adcc1c8c9ec08d7b3bd0e0866d2753659a` |
| Artifact generation source | `e2e0b0adcc1c8c9ec08d7b3bd0e0866d2753659a` |
| First complete seven-file bundle | `09f0b5f1aa75007d762f06e12fb77c36121e2179` |
| Final tested and normalized Stage 3 closure | `cdc6a0070ac70feaa4fa19ba33cd131d69beaadd` |

[Certain] `cdc6a007…` is the authoritative Stage 3 exit snapshot. The intervening commits changed CI, provenance, checksum normalization and documentation—not frozen scientific outcomes.

## 3. Scientific evidence accepted and frozen

[Certain] The following Stage 3 results are approved:

### KernelSHAP

- Seeds: `0,1,2,3,4`
- Mean cosine: `1.0000`
- Mean Spearman: `1.0000`
- Mean nonzero-feature sign agreement: `1.0000`
- All five seeds passed all frozen gates
- MAE, maximum absolute error, top-k agreement and runtime retained as descriptive evidence

### LIME

- Seeds: `0,1,2,3,4`
- Mean cosine: `0.9987`
- Mean sign agreement: `1.0000`
- Mean top-k signal recall: `1.0000`
- Descriptive Spearman: `0.9759`
- Kernel width: `2.4495`
- All five seeds passed all applicable frozen gates

### Determinism

[Certain] Same-seed repeatability passed for:

- RF + KernelSHAP
- RF + LIME
- XGBoost + KernelSHAP
- XGBoost + LIME

[Certain] Different-seed variation remains descriptive and does not constitute a failure.

## 4. Coverage amendment decision

[Certain] The coverage-policy amendment is ratified with these boundaries:

- The 80% threshold remains mandatory.
- Only standalone validation runners may be omitted.
- Reusable metric calculations and scientific logic must remain covered.
- The omission does not automatically apply to future Stage 4 validation modules.
- Stage 4 must reassess coverage after migrating its metrics to Option B+.
- The amendment must disclose that the policy was introduced following a coverage failure and later reviewed and ratified.

## 5. Nonblocking audit note

[Likely] The matching Ubuntu test-ID hash is credible because both platforms used the same commit, lockfile, test count and normalized collection algorithm.

[Certain] Before a paper submission or public archival release, preserve the Ubuntu hash as downloadable CI output or a retained workflow artifact. Do not rely permanently on a statement that the private CI log cannot be accessed.

This is a publication-packaging requirement and does **not** block Stage 4.

## 6. Stage 4 authorization

[Certain] Stage 4 may resume immediately, but the premature Stage 4 commits are not automatically approved.

Antigravity must:

1. Start from the accepted closure snapshot:
    
    ```
    cdc6a0070ac70feaa4fa19ba33cd131d69beaadd
    ```
    
2. Review the quarantined commits:
    
    ```
    5d327f8e9ba9dafff8387a006d1f71701ea633e9
    7684dea32d315f25f170924e4914174668927b86
    ```
    
3. Cherry-pick or reimplement only code consistent with the current Option B+ architecture.
4. Migrate these quarantined metrics to typed immutable contexts:
    - K90 sparsity
    - Runtime
    - Spearman stability
    - Cosine stability
5. Remove all four quarantined first-party `# type: ignore[override]` suppressions.
6. Preserve append-only Stage 3 artifacts.
7. Keep Stage 4 outputs clearly labelled:
    
    ```
    infrastructure-pilot
    ```
    
8. Create a separate Stage 4 artifact directory.
9. Submit a Stage 4 implementation plan before conducting its exit-gate validation.

## 7. Restrictions that remain active

[Certain] This authorization does **not** permit:

- Confirmatory experiments
- Confirmatory real-dataset benchmark execution
- Hypothesis changes
- Threshold changes
- Seed changes
- Post-result metric selection
- Overwriting Stage 3 artifacts
- Treating the AI assistant as an author
- Claiming publication-ready empirical conclusions

[Certain] Confirmatory execution remains blocked until:

1. Stage 4 passes its exit gate.
2. The complete protocol is frozen.
3. Dataset preprocessing and exclusion rules are frozen.
4. The statistical analysis plan is frozen.
5. The preregistration package is approved.
6. External preregistration status is recorded.

## 8. Paste-ready instruction for Antigravity

> DR-007 APPROVED: Stage 3 is closed at final snapshot `cdc6a0070ac70feaa4fa19ba33cd131d69beaadd`. Stage 4 may resume as an infrastructure pilot. Begin from that snapshot. Review the quarantined commits `5d327f8e9ba9dafff8387a006d1f71701ea633e9` and `7684dea32d315f25f170924e4914174668927b86`; do not treat them as automatically accepted. Migrate K90 sparsity, runtime, Spearman stability and cosine stability to the approved Option B+ immutable metric-context interface and remove their first-party override suppressions. Preserve all Stage 3 artifacts unchanged and append-only. Do not run confirmatory experiments, real-dataset confirmatory benchmarks or modify frozen scientific decisions. First submit a restricted Stage 4 implementation plan containing scope, metric contracts, tests, validation evidence, artifact design, provenance strategy and exit criteria.
> 

## Final status

| Gate | Status |
| --- | --- |
| --- | ---: |
| Phase 0 | ✅ Complete |
| Phase 1 Stage 1 | ✅ Complete |
| Phase 1 Stage 2 | ✅ Complete |
| **Phase 1 Stage 3** | **✅ Approved and closed** |
| **Phase 1 Stage 4 development** | **✅ Authorized to resume** |
| Stage 4 exit approval | ⏳ Pending |
| Confirmatory experiments | ⛔ Blocked |
| External preregistration | ⏳ Pending |

[Certain] **DR-006 remediation is closed. No further Stage 3 correction cycle is required unless later evidence reveals an actual reproducibility or scientific defect.**