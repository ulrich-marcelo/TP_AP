"""
Módulo principal de modelos, entrenamiento, datos y evaluación
para la predicción de varianza de Bitcoin.
"""

from .models import (
    VolatilidadGRU,
    VolatilidadLSTM,
    VolatilidadCNN
)

from .data_loader import (
    load_processed_data,
    load_processed_data_with_dates
)

from .trainer import (
    QLIKELoss,
    create_sequences_with_dates,
    train_fold,
    evaluate_fold_predictions,
    run_walk_forward,
)

from .evaluation import (
    diebold_mariano_test,
    analyze_covid_regime
)

__all__ = [
    # Modelos
    "VolatilidadGRU",
    "VolatilidadLSTM",
    "VolatilidadCNN",
    # Cargador de datos
    "load_processed_data",
    "load_processed_data_with_dates",
    # Entrenamiento y Pérdida
    "QLIKELoss",
    "create_sequences",
    "create_sequences_with_dates",
    "train_fold",
    "evaluate_fold_predictions",
    "run_walk_forward",
    "run_walk_forward_sweep",
    # Evaluación Estadística
    "diebold_mariano_test",
    "analyze_covid_regime"
]