[Certain] The plan is implementable, but approving it unchanged would reintroduce weak typing, ambiguous provenance, and the same cross-platform evidence problem that delayed Stage 3.

# DR-008 — Stage 4 Infrastructure Plan Approval

**Project:** EC-TABULAR-001  

**Date:** 2026-07-22  

**Decision:** ✅ **Approved with mandatory corrections**  

**Authorized work:** Stage 4 infrastructure migration only  

**Confirmatory experiments:** ⛔ Not authorized  

**Real-dataset benchmarks:** ⛔ Not authorized  

**Stage 4 scientific exit validation:** ⛔ Requires a separate approved plan

## 1. Approved scope

[Certain] Antigravity may implement:

- `K90Sparsity → BaseMetric[SparsityContext]`
- `RuntimeMetric → BaseMetric[RuntimeContext]`
- `CosineStability → BaseMetric[StabilityContext]`
- `SpearmanStability → BaseMetric[StabilityContext]`
- Removal of the four quarantined first-party override suppressions
- Context validation and contract tests
- Coverage-policy disclosure required by DR-007
- Stage 4 infrastructure artifact scaffolding
- Windows and Ubuntu engineering verification

[Certain] The implementation must begin from:

```
cdc6a0070ac70feaa4fa19ba33cd131d69beaadd
```

The premature Stage 4 commits may be inspected, but they must not be cherry-picked wholesale without reviewing their diffs.

## 2. Mandatory typing correction

[Certain] These proposed declarations are not acceptable under the Option B+ contract:

```python
attributions: list
```

They discard the element type and conflict with the purpose of the generic metric interface.

Use explicit types:

```python
class SparsityContext(BaseMetricContext):
    attributions: tuple[AttributionRecord, ...]
    threshold: float = 0.90
```

```python
class RuntimeContext(BaseMetricContext):
    attributions: tuple[AttributionRecord, ...]
```

A typed immutable tuple is preferred over a mutable list because a frozen Pydantic model does not necessarily make nested lists immutable.

If compatibility requires accepting lists as input, use a pre-validation coercion:

```python
@field_validator("attributions", mode="before")
@classmethod
def coerce_attributions(
    cls,
    value: Sequence[AttributionRecord],
) -> tuple[AttributionRecord, ...]:
    records = tuple(value)
    if not records:
        raise ValueError("attributions must not be empty")
    return records
```

Do not introduce `Any`, bare `list`, `# noqa: ANN401`, or new type suppressions to make the migration pass.

## 3. Context-contract requirements

[Certain] The contexts must preserve existing scientific behavior rather than silently moving metric semantics into validation.

### `SparsityContext`

It must validate:

- Non-empty attributions
- Correct attribution-record type
- Finite threshold
- `0 < threshold <= 1`
- Immutable stored collection

The threshold remains frozen at `0.90`. This implementation does not authorize changing the K90 definition.

### `RuntimeContext`

It must validate:

- Non-empty attributions
- Correct attribution-record type
- Immutable stored collection

[Likely] Validation of individual `runtime_ms` values should remain in the metric if the existing metric converts missing, negative or non-finite runtimes into structured failure records. Do not move such cases into context-construction errors if doing so changes established behavior.

### `StabilityContext`

Before reusing it for cosine and Spearman metrics, verify that it represents all inputs required by both metrics without optional-field misuse.

If cosine and Spearman require a different input shape from Top-k Jaccard, create a dedicated immutable context such as:

```python
PairwiseStabilityContext
```

Do not force unrelated stability metrics into one broad context merely to reduce the number of classes.

## 4. Required tests

[Certain] The proposed 11 tests are approved, with these additions:

- Reject non-finite sparsity thresholds:
    - `NaN`
    - positive infinity
    - negative infinity
- Verify nested collection immutability, not only model-field reassignment.
- Reject an invalid attribution element type.
- Preserve existing handling of:
    - Zero attribution mass
    - Missing runtime
    - Negative runtime
    - Non-finite runtime
    - Constant attribution vectors
    - Undefined Spearman correlation
    - Mismatched vector lengths
- Verify returned `MetricResult` and `FailureRecord` schemas.
- Verify no new first-party suppressions were introduced.

[Certain] `≥166 tests` is a planning estimate, not the principal gate. The real gate is preservation of all existing test IDs plus the approved new contract tests.

## 5. Suppression acceptance criterion

Replace:

> Mypy must report 0 issues, 0 suppressions.
> 

with:

> Mypy must report zero issues. The four quarantined Stage 4 `override` suppressions must be removed. No new first-party suppressions may be introduced.
> 

[Certain] Existing documented project-level suppressions are outside this migration unless the modified code relies on them to hide an interface incompatibility.

Add an explicit audit command, adapted for the available shell:

```bash
rg -n "type:\s*ignore|noqa" \
  src/explaincheck/metrics/sparsity \
  src/explaincheck/metrics/runtime \
  src/explaincheck/metrics/stability \
  src/explaincheck/metrics/contexts.py
```

The report must classify every match; it must not merely report a count.

## 6. Artifact and provenance correction

[Certain] Do not create a mutable `run-manifest.json` stub that will later be overwritten. That conflicts with the append-only requirement.

Use:

```
artifacts/pilot/stage4-infrastructure-v1/
├── README.md
├── infrastructure-validation.json
├── test-inventory.json
├── suppression-audit.json
├── run-manifest.json
└── artifact-checksums.json
```

Use two commits:

1. `stage4InfrastructureSourceCommit`
    - Context migration
    - Metric migration
    - Tests
    - Coverage amendment
    - Validation generator
2. `stage4InfrastructureArtifactCommit`
    - Generated infrastructure evidence
    - Manifest
    - Checksums

The manifest must reference the full source commit. It must not attempt to include its own artifact-commit hash.

[Certain] The approval record for this implementation plan is **DR-008**, not DR-007. DR-007 authorized Stage 4 to resume; DR-008 approves this specific plan.

Use:

```json
{
  "schema_version": "1.0",
  "phase": "stage4-infrastructure-pilot",
  "decision_record": "DR-008",
  "authorized_snapshot": "cdc6a0070ac70feaa4fa19ba33cd131d69beaadd",
  "source_commit": "<full-stage4-source-commit>",
  "status": "infrastructure-validation",
  "confirmatory_experiments": "NOT_AUTHORIZED"
}
```

## 7. Cross-platform verification

[Certain] Windows and Ubuntu must validate the same source commit and lockfile with identical commands:

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/explaincheck/
uv run python scripts/compute_test_id_hash.py
uv run pytest tests/ -q
uv run pytest tests/ --cov=src/explaincheck --cov-report=term-missing
uv run explaincheck validate-stage3-artifacts
```

The exit report must include:

- Full source commit
- `uv.lock` SHA-256
- Python version
- Test count
- Test-ID SHA-256
- Pass/fail/skip counts
- Coverage percentage
- Ruff result
- Mypy result
- Stage 3 artifact-validation result
- Empty `git status --porcelain`

The Ubuntu test-ID hash must be captured as actual workflow output or an uploaded artifact—not inferred from a successful step.

## 8. Revised infrastructure exit criteria

Stage 4 infrastructure development is complete only when:

1. All four Stage 4 override suppressions are removed.
2. No new first-party suppression hides contract incompatibility.
3. All contexts use concrete attribution element types.
4. Nested attribution collections are immutable.
5. Existing scientific formulas and expected values are unchanged.
6. Existing tests and approved context tests pass.
7. Mypy reports zero issues.
8. Ruff lint and format pass.
9. Coverage remains above 80% under the ratified policy.
10. Windows and Ubuntu produce matching test counts and test-ID hashes.
11. Both platforms validate the frozen Stage 3 artifacts.
12. The coverage amendment contains the DR-007 disclosure.
13. Stage 4 evidence is written to `stage4-infrastructure-v1/`.
14. Source and artifact commits are separated.
15. The final working tree is clean.

## 9. Authorization boundaries

[Certain] This approval does not authorize:

- New scientific thresholds
- Metric-definition changes
- New experiment seeds
- Real-dataset execution
- Confirmatory runs
- Stage 3 artifact modification
- Stage 4 scientific exit claims
- Publication-result claims

Any discovery that requires changing K90, runtime, cosine or Spearman semantics must stop implementation and be submitted as a separate scientific decision.

## Paste-ready instruction for Antigravity

> DR-008 APPROVED WITH CORRECTIONS. Implement the Stage 4 infrastructure migration from snapshot `cdc6a0070ac70feaa4fa19ba33cd131d69beaadd`. Use concretely typed immutable attribution collections; do not use bare `list`, `Any`, new `noqa` markers or new type suppressions. Verify that the existing `StabilityContext` correctly represents cosine and Spearman inputs; otherwise introduce a dedicated typed pairwise-stability context. Preserve all metric formulas, fixtures and expected results. Remove the four quarantined `# type: ignore[override]` suppressions. Record evidence in `artifacts/pilot/stage4-infrastructure-v1/` using separate source and artifact commits. Run identical Windows and Ubuntu validation on the same source commit and capture the actual test-ID SHA-256 on both platforms. Do not run confirmatory experiments, real-dataset benchmarks or Stage 4 scientific exit validation. Submit the restricted infrastructure exit report for review.
> 

[Certain] **Execution may begin under DR-008.**