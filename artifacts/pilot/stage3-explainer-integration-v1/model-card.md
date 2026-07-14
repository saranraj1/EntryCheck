# Model Card — Synthetic Linear Logistic Pilot (Stage 2)

## Status
Infrastructure-pilot artifact (Stage 2). Not confirmatory. Not suitable for deployment or substantive decision-making.

## Run ID
`ec-450586ee`

## Model details
- Model: custom L2-regularized logistic regression fitted by deterministic full-batch gradient descent
- Data: synthetic independent Gaussian features
- True coefficients: `[1.5, -1.2, 0.9, -0.7, 0.0, 0.0, 0.0, 0.0]`
- Intercept: -0.15
- Seeds: `[11, 23, 37, 41, 53, 67, 71, 83, 97, 101]`

## Intended use
Validate ExplainCheck Stage 2 package contracts, metric implementations, provenance, and report generation. Migrated from Phase 0.

## Predictive performance across seeds
- Accuracy: 0.797 ± 0.018
- ROC AUC: 0.878 ± 0.015
- Coefficient cosine similarity to generating coefficients: 0.998 ± 0.001

## Explanation methods
- Exact linear attribution: learned coefficient × deviation from training mean (control reference)
- Randomized negative control: within-sample permutation of exact attribution values (negative control)

## Metrics
- Deletion fidelity AOPC@3: mean cumulative absolute logit change after masking top-ranked features to training mean
- Stability Top-3 Jaccard: overlap of top features after small Gaussian perturbation (σ=0.05), restricted to prediction-preserving pairs

## Phase 0 reproduction
{
  "roc_auc_mean": {
    "got": 0.877836,
    "reference": 0.878,
    "tolerance": 0.005,
    "passed": true,
    "delta": -0.000164
  },
  "coef_cosine_mean": {
    "got": 0.998026,
    "reference": 0.998,
    "tolerance": 0.002,
    "passed": true,
    "delta": 2.6e-05
  },
  "exact_fidelity_n200": {
    "got": 1.763635,
    "reference": 1.764,
    "tolerance": 0.02,
    "passed": true,
    "delta": -0.000365
  },
  "neg_fidelity_n200": {
    "got": 0.761285,
    "reference": 0.753,
    "tolerance": 0.03,
    "passed": true,
    "delta": 0.008285
  },
  "exact_stability_n200": {
    "got": 0.953173,
    "reference": 0.957,
    "tolerance": 0.015,
    "passed": true,
    "delta": -0.003827
  },
  "neg_stability_n200": {
    "got": 0.266453,
    "reference": 0.259,
    "tolerance": 0.02,
    "passed": true,
    "delta": 0.007453
  },
  "_all_passed": true
}

## Limitations
This artifact validates Stage 2 machinery. It is not evidence that ExplainCheck outperforms Quantus or OpenXAI, and is not a confirmatory result.
