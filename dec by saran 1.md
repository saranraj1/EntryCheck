# Section 7 — Researcher Decision Record

**Decision date:** 2026-07-14  

**Decision authority:** Saranraj U  

**Decision ID:** `DR-001-PHASE1`  

**Status:** Phase 1 infrastructure work approved; confirmatory experiments not yet approved.

## 1. External preregistration

**Decision: APPROVED — use OSF Registrations.**

Prepare the final registration package using:

- `PROTOCOL_V1.0.md`
- `PREREGISTRATION_READY.md`
- `COMPARISON.md`
- `DATASET_REVIEW.md`
- `PUBLICATION_STRATEGY.md`
- Repository commit hash
- `SHA256SUMS.txt`
- Environment specification
- AI-assistance disclosure

Use an immutable OSF registration and record its URL/DOI in:

- `README.md`
- `CITATION.cff`
- `PROTOCOL_V1.0.md`
- Paper Methods section
- Run manifests

**Important:** Phase 1 infrastructure development may begin before OSF registration, but no confirmatory experiment may begin until the registration is externally timestamped.

## 2. Gradient-boosted-tree implementation

**Decision: APPROVED — use XGBoost.**

Reasons:

- Mature TreeSHAP compatibility
- Widely used in tabular ML research
- Strong reproducibility and CPU support
- Easier comparison with Logistic Regression and Random Forest
- Less framework-specific complexity than CatBoost
- Better fit for the first paper than introducing multiple boosting libraries

Requirements:

- Use `XGBClassifier`
- CPU execution only for the confirmatory benchmark
- Set `n_jobs=1` during controlled runtime measurements
- Pass every random seed explicitly
- Disable nondeterministic GPU implementations
- Record all hyperparameters
- Pin the exact installed stable version in `uv.lock`
- Do not upgrade XGBoost after the pilot environment is frozen
- Record XGBoost, SHAP, NumPy, Python, compiler, and operating-system versions in every run manifest

Use Python **3.12** for Phase 1 rather than Python 3.13 to reduce compatibility risk across SHAP, LIME, XGBoost, and scientific Python packages.

Use `uv` with:

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- A reproducible container definition

## 3. Modular package architecture

**Decision: APPROVED.**

Use this structure:

```
explaincheck/
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── AGENTS.md
├── CITATION.cff
├── LICENSE
├── Dockerfile
├── configs/
│   ├── protocol-v1.yaml
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── confirmatory.yaml
├── src/explaincheck/
│   ├── contracts/
│   ├── config/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   ├── explainers/
│   ├── metrics/
│   │   ├── fidelity/
│   │   ├── stability/
│   │   ├── sparsity/
│   │   ├── runtime/
│   │   ├── correlation/
│   │   ├── missingness/
│   │   └── subgroup/
│   ├── stressors/
│   │   ├── correlation/
│   │   └── missingness/
│   ├── statistics/
│   ├── provenance/
│   ├── reporting/
│   ├── model_cards/
│   └── cli/
├── experiments/
│   ├── manifests/
│   ├── pilots/
│   └── confirmatory/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── golden/
│   └── scientific/
├── artifacts/
│   ├── pilot/
│   ├── exploratory/
│   └── confirmatory/
├── paper/
│   ├── manuscript/
│   ├── tables/
│   ├── figures/
│   └── references/
└── .github/workflows/
    ├── quality.yml
    ├── tests.yml
    ├── scientific-smoke.yml
    └── artifact-validation.yml
```

### Architecture rules

- Use typed interfaces and Pydantic contracts.
- Keep raw results append-only.
- Never overwrite confirmatory artifacts.
- Keep pilot, exploratory, and confirmatory outputs separate.
- Generate paper numbers directly from frozen artifacts.
- Record hashes for code, configuration, data, models, and environment.
- Keep local and global explanation pipelines separate.
- Every metric must declare its direction, range, assumptions, parameters, and aggregation method.

## 4. Phase 1 execution plan

**Decision: APPROVED.**

Proceed in this order:

### Stage 1 — Foundation

Implement:

- Python package skeleton
- Typed configuration system
- Dataset, model, explainer, and metric contracts
- Logging and failure records
- Hashing and provenance
- JSON Schema
- CLI skeleton
- CI quality checks

**Exit gate:** package installs cleanly and all structural tests pass.

### Stage 2 — Synthetic scientific baseline

Migrate the Phase 0 pilot into the package.

Implement:

- Linear synthetic generator
- Exact-linear attribution
- Randomized negative control
- Logistic Regression adapter
- Fidelity AOPC
- Top-k Jaccard stability
- Paper and model-card generation

**Exit gate:** Phase 0 results reproduce within declared numerical tolerance.

### Stage 3 — Explainer integrations

Implement:

- TreeSHAP
- KernelSHAP
- LIME

Record:

- Background dataset
- Sampling budget
- Target class
- Kernel width
- Discretization settings
- Seeds
- Library versions

**Exit gate:** each explainer passes shape, determinism, failure, and analytic-reference tests where applicable.

### Stage 4 — Core metrics

Implement:

- Fidelity AOPC
- Cosine stability
- Spearman stability
- Top-k Jaccard stability
- k90 sparsity
- Runtime and memory metrics

Cross-check at least one overlapping metric against Quantus or OpenXAI.

**Exit gate:** hand fixtures, property tests, golden tests, and external cross-check pass.

### Stage 5 — Stressors

Implement:

- Correlation levels: `0.3`, `0.6`, `0.9`
- Individual attribution agreement
- Grouped attribution conservation
- MCAR and MAR
- Missingness levels: `5%`, `10%`, `20%`, `30%`
- Median/mode, KNN, iterative, native, and indicator-based handling
- Prediction-preservation filtering
- Perturbation rejection-rate reporting

**Exit gate:** controlled synthetic expectations pass.

### Stage 6 — Dataset adapters

Implement in this order:

1. WDBC
2. German Credit
3. Adult
4. Bank Marketing
5. Spambase only as reserve

Bank Marketing must exclude `duration` from the primary model.

**Exit gate:** licenses, DOI, hashes, feature lineage, preprocessing isolation, and datasheets are validated.

### Stage 7 — Statistics and reporting

Implement:

- Stratified bootstrap
- Effect sizes
- Holm correction
- Mixed-effects input generation
- Non-parametric fallback
- Tidy CSV
- Benchmark JSON
- Model cards
- LaTeX tables
- PDF figures
- Paper macros and snippets

**Exit gate:** generated values match frozen results without manual transcription.

### Stage 8 — Infrastructure pilot

Run only:

- Synthetic experiments
- Small smoke subsets from real datasets
- Reduced seeds explicitly marked `infrastructure-only`
- No hypothesis testing
- No confirmatory interpretation

Produce an infrastructure-readiness report.

## 5. Frozen-scope changes

**Decision: NO frozen-scope changes are currently approved**, except this pre-registration wording correction:

Replace:

> H1: No single explainer dominates all four primary dimensions.
> 

With:

> **H1: No single explainer dominates across all seven prespecified evaluation dimensions.**
> 

Reason: the project defines seven evaluation dimensions, so “four” is internally inconsistent.

Because external preregistration has not occurred, make this correction now and record it in:

```
amendments/2026-07-14-pre-registration-clarification.md
```

Mark it as:

- Pre-registration clarification
- No confirmatory outcomes inspected
- No change to datasets, methods, metrics, seeds, or analysis
- Approved by Saranraj U

No other hypothesis, seed, dataset, model family, explainer, metric, stress level, or exclusion rule may be changed without a new approval record.

## 6. Confirmatory run start

**Decision: NOT APPROVED YET.**

Approval is withheld until all these conditions are satisfied:

- OSF registration is externally timestamped.
- Protocol and repository hashes are recorded.
- Python environment and dependencies are locked.
- All unit, property, integration, golden, and scientific tests pass.
- Phase 0 results reproduce from the new package.
- At least one metric is cross-validated against Quantus or OpenXAI.
- Dataset licenses, versions, and hashes are verified.
- Feature-lineage and preprocessing-isolation tests pass.
- Prediction-preservation filtering is validated.
- Infrastructure pilot completes without unresolved critical failures.
- Antigravity produces a confirmatory-run readiness report.
- Saranraj U gives a separate explicit approval after reviewing that report.

Until then, only infrastructure, synthetic validation, smoke tests, documentation, and preregistration preparation are authorized.

---

## Final authorization to Antigravity

> Proceed with Phase 1 infrastructure development according to `DR-001-PHASE1`. Use XGBoost as the frozen boosted-tree implementation, Python 3.12, `uv`, typed modular contracts, and the approved directory structure. Apply the H1 wording clarification and record it as a pre-registration amendment. You may run synthetic, golden, and clearly labelled infrastructure smoke tests. Do not begin confirmatory experiments, publish scientific claims, or change any other frozen research decision without further approval.
>