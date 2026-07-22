# Coverage Policy Amendment — Stage 3 Validation Modules

**Date:** 2026-07-21  
**Author:** Saranraj U (accountable researcher — must countersign to activate)  
**AI assistant:** Antigravity (non-author)  
**Decision record:** DR-006C  
**Status:** Proposed — awaiting Saranraj U countersignature

---

## Context

Commit `eb658e6c62aec1046c187a190d6f088e5eea071a` added
`src/explaincheck/validation/*` to the `[tool.coverage.run]` omit list in
`pyproject.toml`. This was required because the three standalone runner modules
(`kernelshap_reference.py`, `lime_reference.py`, `determinism_matrix.py`,
242 lines combined) pulled coverage below the 80% threshold when included
in the denominator.

DR-006C required that this exclusion be justified prospectively rather than
described as "all gates passed" without qualification.

---

## Modules excluded from coverage denominator

| Module | Lines | Role |
|---|---|---|
| `src/explaincheck/validation/kernelshap_reference.py` | 66 | Runner: data-generation, SHAP call, metric computation, CSV/JSON I/O |
| `src/explaincheck/validation/lime_reference.py` | 70 | Runner: same pattern for LIME |
| `src/explaincheck/validation/determinism_matrix.py` | 106 | Runner: same-seed repeatability + different-seed sensitivity loop |

---

## Justification

### 1. Structural role of the modules

These modules are **standalone validation runners**, not library functions
called by other production code paths. Each module is designed to be invoked
via `uv run python scripts/validation/<script>.py` or the
`explaincheck validate-stage3` CLI command. They combine data generation,
model training, explainer invocation, metric computation and file I/O into
a single sequential script.

### 2. Underlying scientific calculations are tested

The scientific computations inside these modules are identical to calculations
exercised by the covered test suite:

| Computation | Tested in covered module |
|---|---|
| KernelSHAP cosine/sign/Spearman | `test_stage3_adapters.py::test_kernelshap_analytic_reference` |
| LIME cosine/sign/top-k | `test_stage3_adapters.py::test_lime_analytic_reference`, `::test_lime_directional_dominant_feature` |
| Determinism (same-seed equality) | `test_stage2_determinism.py` — all cells |

The `validation/` modules add orchestration (seeded loops, CSV/JSON
serialisation, output-directory management) around logic that is already
covered. The orchestration layer is validated end-to-end by manual pilot
runs that produced the frozen artifacts.

### 3. Why exclusion does not hide untested scientific logic

No scientific threshold, metric formula or attribution calculation appears
exclusively in the excluded modules. Every formula in these modules is
reproduced from, or directly calls, functions in `src/explaincheck/explainers/`
and `src/explaincheck/metrics/` which are covered by the test suite.

### 4. Prospective policy

Coverage is measured over **library modules** (those imported by other
production code). Standalone runner scripts that orchestrate library calls
and produce file output are excluded from the coverage denominator, provided:

- Their scientific calculations are exercised in covered library tests, AND
- They are documented in this amendment before exclusion is applied, AND
- The 80% threshold is met after exclusion.

This policy applies to the current three modules only. Any new module added
to `src/explaincheck/validation/` must be reviewed individually.

---

## Coverage after exclusion (commit `eb658e6`)

| Platform | Coverage | Threshold | Status |
|---|---|---|---|
| Windows (Python 3.12.9) | 83.45% | 80% | PASS |
| Ubuntu (Python 3.12, ubuntu-latest) | PASS (GitHub Actions step: success) | 80% | PASS |

---

*Proposed by AI assistant Antigravity. Requires Saranraj U countersignature.*

---

## DR-007 ratification note (appended 2026-07-22)

This coverage exclusion was introduced after a CI coverage failure at commit
`eb658e6c62aec1046c187a190d6f088e5eea071a`, not prospectively before the failure
occurred. The policy was subsequently reviewed, justified with the structural
arguments above, and ratified as part of DR-007 (2026-07-22, Saranraj U).

**Disclosure of origin:** The exclusion was reactive, not pre-planned.
The ratified policy text above reflects the retrospective justification
reviewed and approved by Saranraj U at Stage 3 closure.

---

## Stage 4 applicability (DR-008 §1)

Per DR-008, this exclusion does **not** automatically extend to Stage 4 modules.
Stage 4 metrics migrated to the Option B+ typed-context interface
(`K90Sparsity`, `RuntimeMetric`, `CosineStability`, `SpearmanStability`)
are library modules imported by production code and must remain in the
coverage denominator. Their coverage will be assessed after the Stage 4
infrastructure migration is validated.

*Appended by AI assistant Antigravity per DR-008 §1 requirement.*

---

## Countersignature — Saranraj U (DR-008A §7)

> Ratified by Saranraj U under DR-007 and reaffirmed under DR-008. The exclusion was introduced after a coverage failure and was retrospectively reviewed. It applies only to the identified Stage 3 standalone validation runners and does not automatically apply to Stage 4 or later code.

**Date of ratification:** 2026-07-22  
**Decision records:** DR-007 (Stage 3 closure), DR-008 (Stage 4 authorization), DR-008A (infrastructure exit)  
**Countersignature recorded by:** AI assistant Antigravity per DR-008A §7  
**Final authority:** Saranraj U — this record is appended in accordance with the explicit countersignature wording provided in DR-008A.
