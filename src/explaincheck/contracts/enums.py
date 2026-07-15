"""
ExplainCheck typed contracts — core enumerations and base data models.

All pipeline components exchange data through these Pydantic-validated types.
Adding a field here is a schema change and must be versioned.
"""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    """Terminal status of a single pipeline cell."""

    SUCCESS = "success"
    FAILURE = "failure"
    EXCLUDED = "excluded"


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ModelFamily(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class ExplainerName(str, Enum):
    TREE_SHAP = "tree_shap"
    KERNEL_SHAP = "kernel_shap"
    LIME = "lime"
    EXACT_LINEAR = "exact_linear"
    RANDOMIZED_NEGATIVE_CONTROL = "randomized_negative_control"


class ExplainerType(str, Enum):
    LOCAL = "local"
    CONTROL_REFERENCE = "control_reference"
    CONTROL_NEGATIVE = "control_negative"
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    LIME = "lime"


class MetricFamily(str, Enum):
    FIDELITY = "fidelity"
    STABILITY = "stability"
    SPARSITY = "sparsity"
    RUNTIME = "runtime"
    CORRELATION = "correlation"
    MISSINGNESS = "missingness"
    SUBGROUP = "subgroup"


class StabilityVariant(str, Enum):
    COSINE = "cosine_similarity"
    SPEARMAN = "spearman_rank_correlation"
    TOP_K_JACCARD = "top_k_jaccard"


class MissingnessMechanism(str, Enum):
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"  # exploratory only


class MissingnessHandler(str, Enum):
    MEDIAN_MODE = "median_mode"
    KNN = "knn"
    ITERATIVE = "iterative"
    NATIVE = "native"
    INDICATOR = "indicator"


class PredictionPreservationStatus(str, Enum):
    PRESERVED = "preserved"
    NOT_PRESERVED = "not_preserved"
    NOT_APPLICABLE = "not_applicable"


class RunLabel(str, Enum):
    SMOKE = "smoke"
    PILOT = "pilot"
    INFRASTRUCTURE_PILOT = "infrastructure-pilot"
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"
