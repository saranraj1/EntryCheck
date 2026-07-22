# DR-006D — Final Review of DR-006C Closure Note

**Accountable researcher:** Saranraj U  

**Decision:** ⚠️ **Conditional closure—not final approval**  

**Stage 3 scientific evidence:** ✅ **Accepted and frozen**  

**Stage 3 repository exit gate:** ❌ **One final verification remains**  

**Stage 4:** **Remains paused until that verification is reported**

The six engineering corrections are substantially complete. However, the closure note explicitly omits one item required by DR-006C and tests a different commit from the declared final closure commit.

## Accepted decisions

### Coverage-policy amendment

The proposed `amendments/2026-07-21-coverage-policy.md` is **accepted and countersigned as a Stage 3 process decision**, subject to these boundaries:

- The 80% coverage threshold remains unchanged.
- Only standalone validation runners may be excluded.
- Reusable scientific calculations must remain in covered package modules.
- The exclusion must not expand automatically to future validation code.
- Stage 4 must reassess coverage after its metric-context migration.
- The amendment must state that the exclusion was introduced after a coverage failure and subsequently reviewed and ratified; it must not call itself prospective if the exclusion was already applied.

This resolves the coverage-policy blocker without requiring a scientific rerun.

### Other accepted evidence

The following are accepted:

- Clean-tree evidence at `5ac48bfa7cd229de7feb1335b8e296d0976eec1b`.
- Genuine checksum-manifest validation.
- Seven artifact-file checksums.
- Full background-data hashes.
- Provenance-role mapping.
- Identical mypy configuration.
- Windows/Ubuntu success at `49ce8685bd9086622d47132e38b0ec047e548063`.
- All previously accepted KernelSHAP, LIME and determinism results.

## Remaining blocker

### Final commit and test-inventory evidence do not match

The declared final closure commit is:

```
5ac48bfa7cd229de7feb1335b8e296d0976eec1b
```

But Windows and Ubuntu were tested at:

```
49ce8685bd9086622d47132e38b0ec047e548063
```

Additionally, the required Ubuntu test-ID hash is absent:

```
Ubuntu test-ID SHA-256: Run script on Ubuntu to reproduce
```

A reproducibility instruction is not the same as recorded reproducibility evidence. Therefore, Item 4 is incomplete, and “Remaining issues: None” is incorrect.

## Required final verification

Run CI on the actual final closure commit:

```
5ac48bfa7cd229de7feb1335b8e296d0976eec1b
```

On both Windows and Ubuntu, execute:

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/explaincheck/
uv run python scripts/compute_test_id_hash.py
uv run pytest tests/ -q
uv run explaincheck validate-stage3-artifacts
```

Report:

| Required field | Windows | Ubuntu |
| --- | --- | --- |
| Full commit | Must be `5ac48bfa…` | Must be `5ac48bfa…` |
| `uv.lock` SHA-256 | Full hash | Same full hash |
| Test count | Expected 155 | Expected 155 |
| Test-ID SHA-256 | Full hash | Must exactly match Windows |
| Tests | Pass | Pass |
| Ruff lint/format | Pass | Pass |
| Mypy | Pass | Pass |
| Artifact validation | Pass | Pass |

If adding the Ubuntu hash to a tracked report creates another commit, do **not** restart the loop. Treat that report-only commit as a documentation commit and identify:

- `verifiedCodeCommit = 5ac48bfa…`
- `finalDocumentationCommit = <new commit>`

No additional cross-platform run is required for a commit that changes only the report.

## Required clarification

The earlier determinism report described LIME backgrounds as 200 rows, while this note reports 100 rows for the determinism cells. State explicitly that:

- The LIME reference-quality experiment used 200 background rows.
- The four-cell determinism experiment used 100 background rows.

If that is not correct, report the actual configurations from the artifacts. Do not rerun or alter results merely to make the values match.

## Paste-ready instruction for Antigravity

> Perform only the DR-006D final verification. Run the complete quality, test-ID hash, test and artifact-validation commands on Windows and Ubuntu at commit `5ac48bfa7cd229de7feb1335b8e296d0976eec1b`. Record both full test-ID SHA-256 values; they must match. Clarify the 200-row LIME reference background versus the 100-row determinism background. Do not change scientific code, artifacts, seeds, fixtures, thresholds, sampling budgets or metric results. Submit only a compact verification table. Stage 4 remains paused.
> 

## Approval rule

If:

1. both platforms pass at `5ac48bfa…`,
2. both report 155 tests,
3. both test-ID hashes match,
4. artifact validation passes, and
5. the background-size distinction is documented,

then **Stage 3 may be approved immediately without another substantive remediation review**.

Until that evidence is supplied, Stage 3 remains scientifically accepted but administratively unclosed, and Stage 4 remains paused.