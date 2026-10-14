import sys
from pathlib import Path

# Resolución dinámica de rutas
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import wandb
from src.data_loader import load_processed_data
from src.models import VolatilidadGRU
from src.trainer import run_walk_forward_sweep

# Carga del dataset preprocesado
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
X_raw, y_raw = load_processed_data(DATA_PATH)

def sweep_agent_wrapper():
    wandb.init()
    config = wandb.config

    # Alinear los gráficos de loss con 'epoch' en lugar de 'step'
    for f in range(7):
        wandb.define_metric(f"fold_{f}_train_loss", step_metric="epoch")
        wandb.define_metric(f"fold_{f}_val_loss", step_metric="epoch")

    run_walk_forward_sweep(
        model_class=VolatilidadGRU,
        X_raw=X_raw,
        y_raw=y_raw,
        config=config,
        epochs=45,
        patience=10
    )

sweep_config = {
    'name': 'sweep-gru',
    'method': 'bayes',
    'metric': {
        'name': 'avg_cross_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 3e-4,
            'max': 3e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'dropout_rate': {'values': [0.2, 0.3, 0.4, 0.5]},
        'hidden_size': {'values': [16, 24, 32]},
        'timesteps': {'values': [5, 7, 14, 21]},
        'batch_size': {'values': [16, 32, 64]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="prediccion-varianza-btc")
    wandb.agent(sweep_id, function=sweep_agent_wrapper, count=40)