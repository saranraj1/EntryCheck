# ExplainCheck Protocol v1.0

**Status:** Frozen preregistration candidate — 2026-07-14  
**Study ID:** EC-TABULAR-001  
**Accountable researcher:** Saranraj U  
**AI assistant:** IGRIS/Notion AI; not an author.

## Aim
Evaluate local tabular feature-attribution methods across fidelity, prediction-conditioned stability, sparsity, runtime, controlled correlation, induced missingness, and subgroup consistency.

## Confirmatory questions
- RQ1: Do explainer rankings disagree across core metrics?
- RQ2: Are there explainer × model and explainer × dataset interactions?
- RQ3: Does redundancy reduce individual agreement more than grouped conservation?
- RQ4: Does attribution drift increase with missingness and vary by handling method among prediction-preserved samples?
- RQ5: Do aggregate results conceal subgroup differences?
- RQ6: Can CI gates detect explanation regressions missed by predictive checks?

## Fixed scope
- Binary tabular classification.
- Real datasets: Adult, German Credit, Bank Marketing, WDBC; Spambase reserve.
- Synthetic: independent linear, nonlinear interaction, correlated redundancy, irrelevant noise, subgroup shift.
- Models: logistic regression, random forest, gradient-boosted trees.
- Local explainers: TreeSHAP, KernelSHAP, LIME.
- Controls: exact-linear reference and randomized within-instance attribution.
- Global methods remain a separate appendix track.

## Primary metrics
1. Deletion fidelity AOPC with data-aware replacement.
2. Stability: cosine, Spearman, and Top-k Jaccard conditional on prediction preservation.
3. k90 attribution-mass sparsity.
4. Runtime: initialization, P50/P95/P99, memory, throughput.
5. Correlation robustness at rho 0.3, 0.6, 0.9, including group conservation.
6. Missingness sensitivity at 5%, 10%, 20%, 30%.
7. Subgroup gaps, standardized effects, and 95% CIs.

## Missingness
MCAR and MAR are primary; MNAR is exploratory. Compare median/mode, KNN, iterative imputation, native handling, and missing indicators. Keep native and induced missingness separate.

## Repetition
Seeds: 11, 23, 37, 41, 53, 67, 71, 83, 97, 101. Minimum 200 held-out explanations per dataset/model/seed, 20 stability perturbations per sample, and 1,000 stratified bootstrap replicates, subject only to a dated pre-confirmatory compute amendment.

## Statistics
Paired comparisons within shared blocks. Prefer mixed models `metric ~ explainer × model × stress_level + (1|dataset) + (1|seed)`; use Friedman/aligned-rank alternatives when necessary. Apply Holm correction, report effect sizes and uncertainty, and avoid a universal composite score.

## Exclusions
Only corrupted artifacts, deterministic implementation failures with minimal reproducers, or prespecified numerical-invalidity failures may be excluded. Preserve every failure in `failures.csv`. Do not remove outliers for weakening a hypothesis.

## Reproducibility
Immutable configs and snapshots; hashes for code, data, model, environment, and configuration; append-only raw results; generated tables/figures; clean-environment reproduction; DOI archive.

## Freeze boundary
This file is locally frozen for preregistration review. It is not an external preregistration until Saranraj U verifies and deposits it in a timestamped registry. Later changes require dated amendments and confirmatory/exploratory labels.
