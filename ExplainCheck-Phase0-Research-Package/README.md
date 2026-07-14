# ExplainCheck Phase-0 Research Package

This package completes the eight immediate actions from the publication-first protocol as far as they can be completed without external accounts or confirmatory real-dataset runs.

## Completed

1. Quantus and OpenXAI design/metric/claim comparison (`COMPARISON.md`).
2. Paired publication strategy selected (`PUBLICATION_STRATEGY.md`).
3. Dataset license and preliminary datasheet review (`DATASET_REVIEW.md`).
4. Executable synthetic linear ground-truth case (`run_phase0.py`).
5. Hand validation of one fidelity and one stability metric (`manual-metric-validation.json`).
6. End-to-end artifact bundle (`artifacts/pilot-v1/`).
7. Ten-seed compute/variance pilot (`tidy-results.csv`, tables, figures).
8. Locally frozen Protocol v1.0 and preregistration-ready record.

## Pilot summary

- Ten seeds; 3,000 generated samples per seed.
- Evaluation sizes: 50, 100, and 200 explanations per seed.
- Mean model ROC AUC: approximately 0.878.
- At n=200, exact-linear deletion fidelity: 1.764 versus 0.753 for the randomized negative control.
- At n=200, exact-linear Top-3 stability: 0.957 versus 0.259 for the negative control.
- Hand fixture: fidelity AOPC@2 expected/computed = 2.25; stability Jaccard@2 expected/computed = 1.0.
- Pilot elapsed time in this sandbox: approximately 2.6 seconds; this is not a forecast for SHAP/LIME or real datasets.

## Reproduce

```bash
python3 run_phase0.py
```

Required Python packages: NumPy, pandas, and matplotlib.

## Important boundary

The pilot validates the pipeline and metric direction using an analytic reference and a negative control. It does **not** establish results for SHAP, LIME, nonlinear models, or real-world data. `PREREGISTRATION_READY.md` is not externally registered until Saranraj U verifies it and deposits it in a timestamped registry such as OSF.
