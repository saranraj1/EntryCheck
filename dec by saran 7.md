[Certain] The migration may be correct, but this report does not prove cross-platform equivalence or identify the final repository state, so Stage 4 infrastructure cannot be closed yet.

# DR-008A — Stage 4 Infrastructure Exit Review

**Decision:** ❌ **Infrastructure exit not yet approved**  

**Implementation status:** ✅ Provisionally accepted  

**Stage 4 scientific validation:** ⛔ Not authorized  

**Confirmatory experiments:** ⛔ Not authorized  

**Required action:** Narrow provenance and evidence closure only

## Accepted implementation evidence

[Certain] The following are accepted provisionally and must not be reworked without discovering a concrete defect:

- `PairwiseStabilityContext`
- `SparsityContext`
- `RuntimeContext`
- Immutable tuple-based attribution collections
- Removal of the four `# type: ignore[override]` suppressions
- Preservation of runtime edge-case handling inside `compute()`
- Twenty added tests
- Reported 175/175 Windows result
- Reported 86.55% Windows coverage
- Coverage-policy disclosure
- Stage 3 artifact preservation
- No confirmatory or real-dataset execution

## Blocking issues

### 1. Windows and Ubuntu are not demonstrated on the same commit

[Certain] The report names several different repository states:

| Role | Commit |
| --- | --- |
| Source | `7cd0bdb75a6143120c74f5536bc3d8ce36d49623` |
| Ubuntu CI | `7ef6391410876c57597c36e9417cc89aa3a92e90` |
| Artifact | `86c9f31...` |
| Submission HEAD | `bc29bf3` |

The Windows result does not identify its tested commit. The Ubuntu run used `7ef6391…`, while the source commit is `7cd0bdb…`.

A workflow-only change may leave scientific code unchanged, but the exit gate requires both platforms to test one exact repository snapshot.

Run the final verification on the same full commit on both platforms.

### 2. Ubuntu test inventory is not verified

[Certain] This value:

```
sha256:13a27407c6701302994c58f71503b3299a3dd407af8ad5143645a9896fcb2ae1
```

is the digest of the uploaded artifact package—not necessarily the test-ID hash stored inside it.

[Certain] The report asks Saranraj U to download the artifact and perform the comparison. That means the required comparison has not yet been completed by the submitting agent.

The closure report must provide the actual contents of the Ubuntu hash file:

```
count=175
sha256=09a032142370b6a3e84264f0cc3e98e0552431ba0e5b86d79d3a4e3b2324f6a0
```

The Ubuntu workflow should also fail automatically if the produced hash differs from the expected inventory hash.

### 3. Full provenance hashes are missing

[Certain] These identifiers are abbreviated:

- Artifact commit: `86c9f31...`
- Submission HEAD: `bc29bf3`

Provide full 40-character hashes for:

- `stage4InfrastructureSourceCommit`
- `stage4InfrastructureArtifactCommit`
- `finalTestedInfrastructureCommit`
- `submissionDocumentationCommit`, if different

Also state which commit first contains the complete final artifact bundle.

### 4. Stage 4 artifact integrity is incomplete

[Certain] DR-008 required:

```
artifact-checksums.json
```

The report lists only:

- `README.md`
- `run-manifest.json`
- `infrastructure-validation.json`
- `test-inventory.json`
- `suppression-audit.json`

The checksum manifest is missing from the reported bundle.

Create a Stage 4 artifact validator that:

- Loads `artifact-checksums.json`
- Recomputes all listed hashes
- Validates required JSON fields and types
- Detects missing and unexpected files
- Excludes the checksum manifest from self-hashing
- Returns a nonzero exit status on failure

Do not modify Stage 3 artifacts.

### 5. The suppression audit contains an unresolved contradiction

[Certain] The report says there are five:

```
# noqa: ANN401
```

markers “on validator return types (`tuple`/`list`).”

`ANN401` concerns dynamically typed `Any`, not ordinary typed tuple/list annotations. The description is therefore insufficient or inaccurate.

The final report must show for each occurrence:

- File
- Line
- Exact function signature
- Whether it existed before DR-008
- Why it is necessary
- Whether `Any` appears in the signature

[Certain] DR-008 prohibited introducing bare `list`, `Any`, or new `ANN401` suppressions in the migration. If any of the five occurrences were newly introduced, replace them with concrete types. Existing unrelated occurrences may remain only if accurately classified.

### 6. Clean-tree evidence is absent

[Certain] The report does not provide final:

```bash
git status --porcelain
```

The output must be literally empty after the final artifact or documentation commit.

### 7. Coverage countersignature must be recorded

[Certain] The policy itself was already accepted through DR-007 and DR-008. The remaining action is administrative: record Saranraj U’s countersignature in the amendment.

Use wording equivalent to:

> Ratified by Saranraj U under DR-007 and reaffirmed under DR-008. The exclusion was introduced after a coverage failure and was retrospectively reviewed. It applies only to the identified Stage 3 standalone validation runners and does not automatically apply to Stage 4 or later code.
> 

This does not require reopening the coverage decision.

## Required DR-008A closure sequence

1. Resolve any new `ANN401` or bare-type violations.
2. Add the Stage 4 checksum manifest and validator.
3. Record the coverage-amendment countersignature.
4. Create one final source/artifact closure snapshot.
5. Run identical Windows and Ubuntu checks on that snapshot.
6. Extract and compare the actual Ubuntu test-ID hash.
7. Record full provenance hashes.
8. Verify an empty working tree.
9. Submit a compact closure note.

## Required final verification

Run on both platforms at the same commit:

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/explaincheck/
uv run python scripts/compute_test_id_hash.py
uv run pytest tests/ -q
uv run pytest tests/ --cov=src/explaincheck --cov-report=term-missing
uv run explaincheck validate-stage3-artifacts
uv run explaincheck validate-stage4-infrastructure-artifacts
```

Report:

| Field | Windows | Ubuntu |
| --- | --- | --- |
| Full commit | Exact hash | Same exact hash |
| `uv.lock` SHA-256 | Full hash | Same hash |
| Test count | Expected 175 | Must match |
| Test-ID SHA-256 | Full hash | Must match |
| Test result | Pass | Pass |
| Coverage | >80% | >80% |
| Ruff | Pass | Pass |
| Mypy | Pass | Pass |
| Stage 3 artifacts | 7/7 pass | 7/7 pass |
| Stage 4 artifacts | Pass | Pass |

## Paste-ready instruction for Antigravity

> DR-008A: Stage 4 implementation is provisionally accepted, but infrastructure closure is withheld. Perform only provenance and evidence corrections. Test one exact final commit on Windows and Ubuntu. Extract the actual Ubuntu test-ID hash from the uploaded artifact and compare it with the Windows value; do not report the upload-package digest as the test-ID hash. Provide full 40-character source, artifact, tested and documentation hashes. Add `artifact-checksums.json` and a genuine Stage 4 artifact validator. Audit all five `ANN401` occurrences with exact signatures and remove any introduced bare `list` or `Any` annotations. Record Saranraj U’s countersignature in the coverage amendment and provide literally empty `git status --porcelain`. Do not change metric formulas, contexts, fixtures, thresholds, seeds or scientific results. Do not begin Stage 4 scientific validation.
> 

[Certain] **Do not create DR-009 scientific-pilot scope yet. Close DR-008A first.**