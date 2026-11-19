import argparse
import sys
from pathlib import Path

import pandas as pd
import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import VolatilidadCNN, load_processed_data, run_walk_forward_sweep

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
SWEEP_LOG_DIR = ROOT / "results" / "sweeps"
SWEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = SWEEP_LOG_DIR / "sweep_cnn_history.csv"

TRAIN_WINDOW_DAYS = 1095
INTERNAL_VAL_DAYS = 90
VAL_WINDOW_DAYS = 182
STEP_DAYS = 182

# Carga estricta de Fase A (<= 2025-12-25) para blindar el test ciego de 2026
X_raw, y_raw = load_processed_data(DATA_PATH, phase="phase_a")
N_FOLDS = (len(X_raw) - TRAIN_WINDOW_DAYS) // STEP_DAYS


def sweep_agent_wrapper():
    run = wandb.init()
    config = wandb.config

    for f in range(N_FOLDS):
        wandb.define_metric(f"fold_{f}_epoch")
        wandb.define_metric(f"fold_{f}_train_loss", step_metric=f"fold_{f}_epoch")
        wandb.define_metric(f"fold_{f}_val_loss", step_metric=f"fold_{f}_epoch")

    metrics = run_walk_forward_sweep(
        model_class=VolatilidadCNN,
        X_raw=X_raw,
        y_raw=y_raw,
        config=config,
        epochs=45,
        patience=10,
        seed=42,
        train_window_days=TRAIN_WINDOW_DAYS,
        internal_val_days=INTERNAL_VAL_DAYS,
        val_window_days=VAL_WINDOW_DAYS,
        step_days=STEP_DAYS,
    )

    # Respaldo local estructurado para selección automática Top-2 Mean + Top-2 Median
    clean_cfg = {k: v for k, v in dict(config).items() if not str(k).startswith("_")}
    record = {
        "architecture": "CNN_1D",
        "run_id": run.id if run is not None else "local",
        "run_name": run.name if run is not None else "local",
        **clean_cfg,
        **metrics,
    }
    pd.DataFrame([record]).to_csv(
        HISTORY_CSV,
        mode="a",
        header=not HISTORY_CSV.exists(),
        index=False,
    )


sweep_config = {
    "name": "sweep-cnn-phase-a",
    "method": "bayes",
    "metric": {
        "name": "avg_cross_val_loss",
        "goal": "minimize",
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 3e-4,
            "max": 3e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "dropout_rate": {"values": [0.2, 0.3, 0.4, 0.5]},
        "hidden_size": {"values": [16, 24, 32]},
        "kernel_size": {"values": [3, 5, 7]},  # Kernels impares para padding simétrico
        "num_layers": {"values": [1, 2]},
        "timesteps": {"values": [5, 7, 14, 21]},
        "batch_size": {"values": [16, 32, 64]},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimización Bayesiana W&B para CNN 1D (Fase A)")
    parser.add_argument("--project", type=str, default="prediccion-varianza-btc-2")
    parser.add_argument("--count", type=int, default=40, help="Cantidad de corridas del agente.")
    parser.add_argument("--sweep-id", type=str, default=None, help="ID de sweep existente para continuar.")
    args = parser.parse_args()

    print(f"Iniciando Sweep CNN 1D sobre Fase A ({len(X_raw)} obs | {N_FOLDS} folds)...")
    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, function=sweep_agent_wrapper, count=args.count, project=args.project)
    print(f"Historial local guardado en: {HISTORY_CSV}")