# Quantus–OpenXAI–ExplainCheck comparison

## Review scope
This comparison covers the main design, metrics, experiments, and claims in the Quantus JMLR paper and OpenXAI NeurIPS Datasets and Benchmarks paper. Repository behavior must be rechecked against the exact package versions used during implementation.

## Feature and claim matrix

| Dimension | Quantus | OpenXAI | ExplainCheck v1.0 |
|---|---|---|---|
| Purpose | Quantitative XAI evaluation toolkit | End-to-end benchmark and leaderboards | Tabular stress benchmark plus model cards and CI gates |
| Metrics | 30+ across faithfulness, robustness, localization, complexity, randomization, axiomatic categories | Ground-truth/predictive faithfulness, stability, and group disparities | Fidelity, prediction-conditioned stability, sparsity, runtime, correlation robustness, missingness sensitivity, subgroup consistency |
| Explainers | Accepts generated/precomputed attributions | LIME, SHAP, four gradient methods, random baseline | TreeSHAP, KernelSHAP, LIME, exact-linear reference, randomized negative control |
| Data | User supplied; examples across modalities | SynthGauss plus seven real tabular datasets | Controlled synthetic cases plus four license-reviewed UCI datasets |
| Models | PyTorch/TensorFlow abstraction | Logistic regression and neural networks | Logistic regression, random forest, boosted trees |
| Correlation stress | No dedicated tabular stress ladder in paper | Synthetic design intentionally assumes independence | Prespecified rho ladder plus individual/group conservation |
| Missingness stress | No dedicated mechanism benchmark | No dedicated mechanism/imputation benchmark | MCAR/MAR primary, MNAR exploratory, 5–30%, multiple handlers |
| Stability control | Output approximately unchanged | RIS/RRS/ROS | Explicit prediction-preservation filtering and rejection rate |
| Uncertainty | Sensitivity emphasized | Mean and standard error | Repeated seeds, bootstrap CIs, paired/hierarchical tests, rank uncertainty |
| Outputs | Scores and plots | Tables and leaderboards | Raw/tidy results, JSON Schema, LaTeX, model card, manifest, CI decision |
| Primary claim | Broader standardized XAI quantification | Systematic post-hoc explanation benchmark | Data-imperfection stress testing and evidence-to-governance pipeline |

## Defensible novelty
ExplainCheck must not claim to be the first XAI evaluator, tabular benchmark, synthetic-ground-truth benchmark, or fairness-aware explanation evaluation. Its defensible novelty is the integrated protocol for correlation and missingness stressors, prediction-conditioned explanation drift, uncertainty-aware subgroup reporting, and immutable artifacts that feed model cards and CI regression checks.

## Required comparisons in the paper
1. Reproduce at least one overlapping fidelity metric against Quantus or OpenXAI on identical frozen inputs.
2. Compare stability with an established implementation or analytic fixture.
3. Ablate prediction-preservation filtering.
4. Compare individual-feature with grouped-feature correlation robustness.
5. Report replacement baselines and all metric parameters.

## Sources
- Quantus: https://jmlr.org/papers/v24/22-0142.html
- OpenXAI: https://arxiv.org/abs/2206.11104
