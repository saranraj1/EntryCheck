"""Missingness stressor — induces MCAR/MAR/MNAR missingness at specified levels."""
# Stage 5 implementation: MissingnessStressor
# Levels: 5%, 10%, 20%, 30% (frozen)
# Mechanisms: MCAR, MAR (confirmatory); MNAR (exploratory)
# Handlers: median_mode, knn, iterative, native, indicator
# Native and induced missingness must be kept strictly separate.
