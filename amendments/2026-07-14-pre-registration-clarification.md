# Amendment Record — Pre-Registration Clarification

**Amendment ID:** AMD-001  
**Date:** 2026-07-14  
**Amendment type:** Pre-registration wording clarification  
**Approved by:** Saranraj U (Decision Record DR-001-PHASE1, Section 5)  
**Recorded by:** Antigravity AI assistant (not an author)  

---

## Status flags

- [x] Pre-registration clarification (no change to design, methods, analysis, or scope)
- [x] No confirmatory outcomes inspected at time of amendment
- [x] No change to datasets, models, explainers, metrics, seeds, stress levels, or exclusion rules
- [x] No change to statistical analysis plan
- [x] Approved by Saranraj U before any confirmatory run

---

## Change

### File affected

`PREREGISTRATION_READY.md` — Hypothesis H1

### Original wording

> H1: No single explainer dominates all four primary dimensions.

### Corrected wording

> H1: No single explainer dominates across all seven prespecified evaluation dimensions.

### Rationale

The ExplainCheck protocol defines **seven** primary evaluation dimensions:

1. Deletion fidelity AOPC
2. Prediction-conditioned stability
3. Attribution-mass sparsity (k90)
4. Runtime and resource use
5. Correlation robustness
6. Missingness sensitivity
7. Subgroup consistency

The word "four" in the original wording was an internal inconsistency introduced during drafting. The study design and PROTOCOL_V1.0.md have always specified seven dimensions. This correction aligns H1 with the rest of the protocol without altering any scientific decision.

### Impact assessment

| Dimension | Changed? |
|---|---|
| Datasets | No |
| Models | No |
| Explainers | No |
| Metrics | No |
| Seeds | No |
| Stress levels | No |
| Exclusion rules | No |
| Statistical plan | No |
| Hypotheses H2–H6 | No |
| Research questions RQ1–RQ6 | No |

---

## Amendment provenance

This amendment was prepared by the Antigravity AI assistant and must be verified by Saranraj U before the preregistration package is deposited externally. No external registration has occurred at the time of writing. The corrected wording replaces the original only in the registration package deposited at OSF; the locally frozen Phase 0 file may retain the original text with a pointer to this amendment.
