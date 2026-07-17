# `DR-006A-STAGE3-FINALIZATION-PLAN-APPROVAL`

**Authority:** Saranraj U  

**Decision:** Plan approved with the corrections below  

**Stage 4:** Remains paused and quarantined  

**Confirmatory experiments:** Not approved

## 1. Metric contract decision

**Choose Option B+, not a plain unstructured `TypedDict`.**

Use a generic `BaseMetric` with immutable, typed context models:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

class BaseMetricContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    seed: int
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]

ContextT = TypeVar("ContextT", bound=BaseMetricContext)

class BaseMetric(ABC, Generic[ContextT]):
    @abstractmethod
    def compute(self, context: ContextT) -> "MetricResult":
        ...
```

Define metric-specific contexts:

```python
class AOPCContext(BaseMetricContext):
    inputs: NDArray
    attributions: NDArray
    model: ModelAdapter
    baseline: NDArray
    k_max: int

class StabilityContext(BaseMetricContext):
    original_attributions: NDArray
    perturbed_attributions: NDArray
    original_predictions: NDArray
    perturbed_predictions: NDArray
    k: int
```

Then:

```python
class AOPCMetric(BaseMetric[AOPCContext]):
    def compute(self, context: AOPCContext) -> MetricResult:
        ...

class TopKJaccardMetric(BaseMetric[StabilityContext]):
    def compute(self, context: StabilityContext) -> MetricResult:
        ...
```

### Requirements

- Remove both `# type: ignore[override]` suppressions.
- Do not use one context containing many optional `Any` fields.
- Contexts must be immutable.
- Validate dimensions and feature ordering at context creation.
- Add contract tests proving each metric accepts only its declared context.
- Keep shared metadata in `BaseMetricContext`.
- Use metric-specific subclasses for scientific inputs.

This preserves inheritance while avoiding incompatible method signatures.

---

## 2. Multi-seed gate decision

**Yes. Run the multi-seed experiments and report the results exactly as observed. Do not adjust gates, fixtures, seeds, sampling budgets or aggregation after inspecting the results.**

Use seeds:

```
0, 1, 2, 3, 4
```

### Gate application

Apply already declared gates to the **mean across the five seeds**.

Also report:

- Per-seed result
- Mean
- Sample standard deviation
- Minimum
- Maximum
- Number of seeds passing individually

Do not create new confirmatory gates after observing the results.

### KernelSHAP gates

Apply:

```
Mean cosine similarity ≥ 0.99
Mean Spearman correlation ≥ 0.95
Mean nonzero-feature sign agreement ≥ 0.95
```

Report these without a new threshold:

- Mean absolute error
- Maximum absolute error
- Top-k agreement
- Runtime
- Failure count

Top-k agreement remains descriptive because no KernelSHAP top-k gate was previously frozen.

### LIME gates

Apply:

```
Mean cosine similarity ≥ 0.95
Mean nonzero-feature sign agreement ≥ 0.90
Mean Top-k signal recall ≥ 0.90
```

Report Spearman correlation descriptively because no LIME Spearman threshold was previously frozen.

If any gate fails:

- Record the failure.
- Do not rerun with different settings.
- Do not increase `nsamples` or `num_samples`.
- Do not remove an unfavorable seed.
- Investigate only after preserving the original result artifact.

---

# Required corrections to the implementation plan

## 3. Do not generate research tables from test files

Files under `tests/` should test behavior, not produce canonical research artifacts as side effects.

Use:

```
scripts/validation/stage3_kernelshap_multiseed.py
scripts/validation/stage3_lime_multiseed.py
scripts/validation/stage3_determinism_matrix.py
```

The scripts should generate structured JSON or CSV evidence.

Tests should:

- Call the underlying validation functions
- Assert schemas
- Assert frozen gates
- Verify deterministic reproduction
- Avoid writing into canonical artifact directories

Recommended source location:

```
src/explaincheck/validation/
├── kernelshap_reference.py
├── lime_reference.py
└── determinism_matrix.py
```

## 4. Correct Git-cleanliness sequencing

The tree cannot remain clean while uncommitted remediation is being implemented.

Use these checkpoints:

1. Record the deviation amendment.
2. Remove tracked generated files.
3. Commit the amendment and Git-hygiene correction.
4. Verify a clean tree.
5. Implement the scientific and contract corrections.
6. Create the Stage 3 final source commit.
7. Verify a clean tree.
8. Run validation from that exact source commit.
9. Generate artifacts.
10. Create a separate artifact commit.
11. Verify the final tree is clean.

## 5. Preserve append-only artifacts

Do **not** overwrite:

```
artifacts/pilot/stage3-explainer-integration-v1/
```

That directory is evidence from the earlier Stage 3 attempt.

Write finalization evidence to:

```
artifacts/pilot/stage3-finalization-v1/
```

Include:

```
integration-validation.json
kernelshap-multiseed.csv
kernelshap-multiseed-summary.json
lime-multiseed.csv
lime-multiseed-summary.json
determinism-matrix.json
static-analysis-audit.json
environment-windows.json
environment-ubuntu.json
test-inventory-windows.json
test-inventory-ubuntu.json
process-deviation.md
SHA256SUMS.txt
```

## 6. Use two final commits

Do not combine source and generated artifacts into a self-referential commit.

### Source commit

Contains:

- Amendment
- Git-hygiene changes
- Metric-context refactor
- Adapter/test corrections
- Validation code
- CI changes

Record as:

```
stage3FinalSourceCommit
```

### Artifact commit

Contains:

- Multi-seed results
- Determinism matrix
- Environment records
- Integration validation
- Checksums
- Final report

Record separately as:

```
stage3FinalArtifactCommit
```

Generated artifacts must reference the source commit, not their own artifact commit.

## 7. Correct artifact-generation command

Do not use a one-seed generic pilot command to generate Stage 3 final evidence.

Create a dedicated command, for example:

```bash
uv run explaincheck validate-stage3 \
  --seeds 0 1 2 3 4 \
  --output-dir artifacts/pilot/stage3-finalization-v1
```

Then:

```bash
uv run explaincheck validate-artifacts \
  --dir artifacts/pilot/stage3-finalization-v1
```

The validation command must refuse to overwrite a non-empty frozen artifact directory unless an explicit development-only flag is supplied. That flag must never be used for frozen runs.

## 8. Matching Windows and Ubuntu evidence

Both platforms must test the same full source commit and lockfile hash.

Use identical commands:

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/explaincheck/
uv run pytest tests/ -v --tb=short
```

Required equality:

- Same source commit
- Same `uv.lock` hash
- Same collected test IDs
- Same test markers
- Same validation configuration

Platform-specific skips are allowed only if:

- Explicitly marked
- Justified
- Listed by test ID
- Unrelated to scientific correctness

Do not require bitwise equality for stochastic numerical outputs. Apply declared numerical tolerances.

## 9. Determinism definitions

For each required model–explainer cell, report two separate properties.

### Same-seed repeatability

Verify:

- Same model predictions
- Same feature ordering
- Same background hash
- Same attributions exactly or within a declared tolerance
- Same non-time provenance fields

### Different-seed sensitivity

Verify:

- The seed is recorded.
- A method expected to be stochastic can produce different outputs.
- Differences remain finite and schema-valid.
- Scientific-quality metrics are reported.
- Different outputs are not automatically classified as failures.

Do not write a test that requires different seeds to produce different outputs for a deterministic method.

## 10. Process deviation

Create:

```
amendments/2026-07-15-stage4-premature-start.md
```

Classify it as:

```
Type: Process deviation
Scientific impact: None identified
Confirmatory outcomes inspected: No
Frozen protocol changed: No
Corrective action: Stage 4 paused and commits quarantined
```

List every prematurely created Stage 4 commit using its full hash.

Do not delete or rewrite those commits. Keep them quarantined and exclude them from the final Stage 3 source snapshot unless they are strictly required for the Stage 3 metric-contract correction.

## 11. `preservation_rate`

Wire `prediction_preservation_rate` into:

- Per-run results
- Tidy result schema
- Integration-validation JSON
- Model card or limitations section
- Generated Methods snippet

Remove the `F841` suppression after the value is used.

## 12. Exception tests

Replace broad exceptions such as:

```python
pytest.raises(Exception)
```

with exact expected exceptions:

```python
pytest.raises(ValueError)
pytest.raises(TypeError)
pytest.raises(ValidationError)
pytest.raises(UnsupportedModelError)
pytest.raises(FeatureSchemaError)
```

Define project-specific exception classes where they improve clarity.

Do not globally ignore `B017` after these replacements unless unrelated legacy tests remain; scope any remaining exceptions narrowly.

---

# Final authorization

> The Stage 3 Finalization Plan is approved with Option B+: a generic `BaseMetric` using immutable, metric-specific Pydantic context models. Run KernelSHAP and LIME multi-seed validation using seeds 0–4 and report all outcomes exactly as observed. Apply only the previously declared gates to five-seed means; do not change gates or experimental settings after viewing results. Keep Stage 4 paused, preserve existing artifacts, use separate source and artifact commits, and submit the restricted ten-item finalization report when complete.
>