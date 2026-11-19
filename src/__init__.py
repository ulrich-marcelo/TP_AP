"""
Módulo principal de modelos, entrenamiento, datos y evaluación.
"""

from .models import VolatilidadGRU, VolatilidadLSTM, VolatilidadCNN
from .data_loader import (
    EXPECTED_FEATURE_COLS,
    TARGET_COL,
    PHASE_A_END_DATE,
    load_processed_data,
    load_processed_data_with_dates,
    load_returns_reference,
    load_dataset_metadata,
)
from .trainer import (
    MIN_LOG_VAR,
    MAX_LOG_VAR,
    DEFAULT_EPS,
    set_deterministic_seed,
    log_var_to_var_tensor,
    log_var_to_var_numpy,
    QLIKELoss,
    create_sequences,
    create_sequences_with_dates,
    evaluate_model_qlike,
    train_fold,
    run_walk_forward,
    run_walk_forward_sweep,
)
from .evaluation import (
    compute_qlike_vector,
    get_significance_code,
    diebold_mariano_full,
    diebold_mariano_test,
    validate_predictions_integrity,
    analyze_covid_regime,
)

__all__ = [
    "VolatilidadGRU",
    "VolatilidadLSTM",
    "VolatilidadCNN",
    "EXPECTED_FEATURE_COLS",
    "TARGET_COL",
    "PHASE_A_END_DATE",
    "load_processed_data",
    "load_processed_data_with_dates",
    "load_returns_reference",
    "load_dataset_metadata",
    "MIN_LOG_VAR",
    "MAX_LOG_VAR",
    "DEFAULT_EPS",
    "set_deterministic_seed",
    "log_var_to_var_tensor",
    "log_var_to_var_numpy",
    "QLIKELoss",
    "create_sequences",
    "create_sequences_with_dates",
    "evaluate_model_qlike",
    "train_fold",
    "run_walk_forward",
    "run_walk_forward_sweep",
    "compute_qlike_vector",
    "get_significance_code",
    "diebold_mariano_full",
    "diebold_mariano_test",
    "validate_predictions_integrity",
    "analyze_covid_regime",
]