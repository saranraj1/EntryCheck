# ExplainCheck

**Stress-testing tabular feature attribution under correlation and missingness**

[![Tests](https://github.com/saranraju/ExplainCheck/actions/workflows/tests.yml/badge.svg)](https://github.com/saranraju/ExplainCheck/actions/workflows/tests.yml)
[![Quality](https://github.com/saranraju/ExplainCheck/actions/workflows/quality.yml/badge.svg)](https://github.com/saranraju/ExplainCheck/actions/workflows/quality.yml)

> **Status:** Phase 1 — Infrastructure Development (pre-confirmatory)  
> **Accountable researcher:** Saranraj U  
> **OSF Preregistration:** _Pending external deposit — URL/DOI to be recorded here_  
> **Protocol version:** 1.0 (Study ID: EC-TABULAR-001)

---

## Overview

ExplainCheck is a publication-first benchmark that evaluates local tabular feature-attribution
methods (TreeSHAP, KernelSHAP, LIME) across seven measurement dimensions:

1. **Deletion fidelity** (AOPC with data-aware replacement)
2. **Prediction-conditioned stability** (cosine, Spearman, Top-k Jaccard)
3. **Attribution-mass sparsity** (k90)
4. **Runtime and resource use** (P50/P95/P99 latency, memory, throughput)
5. **Correlation robustness** (ρ ∈ {0.3, 0.6, 0.9}; individual + grouped)
6. **Missingness sensitivity** (MCAR/MAR at 5/10/20/30%; multiple handlers)
7. **Subgroup consistency** (gaps, effect sizes, 95% CIs)

ExplainCheck produces immutable machine-readable benchmark results, model cards,
paper-ready tables and figures, and CI/CD explanation-regression decisions.

### Defensible contribution

ExplainCheck's primary contribution is **not** inventing XAI evaluation. Quantus and OpenXAI
are major predecessors. The defensible contribution is the integrated protocol for correlation
and missingness stressors, explicit prediction-conditioned stability, grouped attribution
analysis, repeated-seed uncertainty, and conversion of frozen research evidence into
model cards and CI quality gates.

---

## Quick start

```bash
# Requires uv (https://docs.astral.sh/uv/)
git clone <repo-url>
cd ExplainCheck
uv sync --extra dev
uv run pytest tests/unit -m unit
uv run explaincheck --help
```

---

## Reproduce the Phase 0 pilot

```bash
uv run explaincheck pilot synthetic --config configs/pilot.yaml --out artifacts/pilot/
```

---

## Project structure

```
explaincheck/
├── pyproject.toml          # Package config and pinned dependencies
├── uv.lock                 # Fully resolved lock file (run `uv lock` to regenerate)
├── .python-version         # Python 3.12
├── configs/                # Frozen experiment configurations (YAML)
├── src/explaincheck/       # Main Python package
│   ├── contracts/          # Typed Pydantic interfaces
│   ├── config/             # Config loading and validation
│   ├── datasets/           # Dataset adapters
│   ├── preprocessing/      # Preprocessing pipelines
│   ├── models/             # Model adapters (LR, RF, XGBoost)
│   ├── explainers/         # Explainer adapters (TreeSHAP, KernelSHAP, LIME)
│   ├── metrics/            # Metric implementations (7 families)
│   ├── stressors/          # Correlation and missingness stressors
│   ├── statistics/         # Statistical analysis (bootstrap, tests, effects)
│   ├── provenance/         # Hashing and artifact provenance
│   ├── reporting/          # Table, figure, and LaTeX generators
│   ├── model_cards/        # Model card generators
│   └── cli/                # CLI entry points
├── experiments/            # Experiment definitions and manifests
├── tests/                  # Test suite (unit/property/integration/golden/scientific)
├── artifacts/              # Frozen run outputs (append-only, never overwrite confirmatory)
├── paper/                  # Manuscript, tables, figures
└── .github/workflows/      # CI/CD pipelines
```

---

## Research scope (frozen — see PROTOCOL_V1.0.md)

| Dimension | Selection |
|---|---|
| Task | Binary tabular classification |
| Datasets | Adult, German Credit, Bank Marketing, WDBC (reserve: Spambase) |
| Models | Logistic Regression, Random Forest, XGBoost (frozen) |
| Explainers | TreeSHAP, KernelSHAP, LIME; controls: exact-linear, randomized |
| Seeds | 11, 23, 37, 41, 53, 67, 71, 83, 97, 101 |
| Missingness (confirmatory) | MCAR, MAR |
| Missingness (exploratory) | MNAR |

---

## Important boundaries

- **Pilot results are pipeline-validation only.** They do not establish findings about SHAP,
  LIME, nonlinear models, or real-world data.
- **No confirmatory run may start** until OSF preregistration is externally timestamped.
- **Raw and confirmatory artifacts are append-only and must not be overwritten.**
- See `PROTOCOL_V1.0.md` and `amendments/` for the complete frozen scope and any approved changes.

---

## AI assistance disclosure

Antigravity (Google DeepMind AI assistant) assisted with code scaffolding, documentation
drafting, and implementation. Saranraj U verified sources, executed and inspected code,
approved all decisions, and accepts full responsibility. The AI is not an author.

---

## Citation

If you use this benchmark, please cite (BibTeX in `CITATION.cff`):

> Saranraj U. _ExplainCheck: Stress-Testing Tabular Feature Attribution under Correlation
> and Missingness._ [Preregistration: OSF — URL pending]. 2026.

---

## License

MIT — see `LICENSE`.
