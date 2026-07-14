# Model Card — Synthetic Linear Logistic Pilot

## Status
Pilot artifact; not suitable for deployment or substantive decision-making.

## Model details
- Model: custom L2-regularized logistic regression fitted by deterministic full-batch gradient descent
- Data: synthetic independent Gaussian features
- True coefficients: `[1.5, -1.2, 0.9, -0.7, 0.0, 0.0, 0.0, 0.0]`
- Seeds: `[11, 23, 37, 41, 53, 67, 71, 83, 97, 101]`

## Intended use
Validate ExplainCheck contracts, metric implementations, provenance, and report generation before adding external explainers or real-world datasets.

## Predictive performance across seeds
- Accuracy: 0.797 ± 0.018
- ROC AUC: 0.878 ± 0.015
- Coefficient cosine similarity to generating coefficients: 0.998 ± 0.001

## Explanation methods
- Exact linear attribution: learned coefficient × deviation from training mean
- Randomized negative control: within-sample permutation of exact attribution values

## Metrics
- Deletion fidelity AOPC@3: mean cumulative absolute logit change after masking top-ranked features to the training mean
- Stability Top-3 Jaccard: overlap of top features after small Gaussian perturbation, restricted to prediction-preserving pairs

## Limitations
This artifact validates machinery; it is not evidence that ExplainCheck outperforms Quantus or OpenXAI and is not a confirmatory result.
