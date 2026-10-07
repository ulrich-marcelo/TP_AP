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

# Carga de datos
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
X_raw, y_raw = load_processed_data(DATA_PATH)

def sweep_agent_wrapper():
    wandb.init()
    config = wandb.config

    # Pasamos VolatilidadGRU como CLASE (sin instanciar)
    run_walk_forward_sweep(
        model_class=VolatilidadGRU,
        X_raw=X_raw,
        y_raw=y_raw,
        config=config,
        epochs=30
    )

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'avg_cross_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Muestreo logarítmico: explora equitativamente órdenes de magnitud (1e-4, 1e-3, etc.)
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 5e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        # Valores discretos limpios para reporte de tesis
        'dropout_rate': {'values': [0.3, 0.5]},
        'hidden_size': {'values': [16, 24, 32]},
        'timesteps': {'values': [3, 7, 10, 14]},
        'batch_size': {'values': [16, 32]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="prediccion-varianza-btc")
    wandb.agent(sweep_id, function=sweep_agent_wrapper, count=20)