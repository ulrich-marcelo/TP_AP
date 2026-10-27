import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Resolución canónica de rutas: añade la raíz del proyecto a sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2. Imports limpios desde el paquete src
from src import (
    VolatilidadCNN,
    VolatilidadGRU,
    VolatilidadLSTM,
    load_processed_data_with_dates,
    run_walk_forward
)

SEEDS = [42, 123, 456, 789, 2024]
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configuraciones derivadas de los Sweeps en W&B
CONFIG_CNN = {
    'timesteps': 7,
    'hidden_size': 24,
    'dropout_rate': 0.3,
    'kernel_size': 3,
    'num_layers': 2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 32
}

CONFIG_GRU = {
    'timesteps': 7,
    'hidden_size': 32,
    'dropout_rate': 0.3,
    'num_layers': 1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 32
}

CONFIG_LSTM = {
    'timesteps': 7,
    'hidden_size': 32,
    'dropout_rate': 0.3,
    'num_layers': 1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 32
}

def execute_multi_seed_experiment(model_name, model_class, config):
    _, X, y, dates, _ = load_processed_data_with_dates(DATA_PATH)
    
    seed_results = []
    all_preds_list = []
    history_dict = {}

    print("\n" + "="*70)
    print(f"INICIANDO EVALUACIÓN MULTISEMILLA: {model_name}")
    print("="*70)
    
    for seed in SEEDS:
        print(f" -> Ejecutando Semilla {seed}...")
        df_metrics, df_preds, histories = run_walk_forward(
            model_class=model_class,
            X_raw=X,
            y_raw=y,
            dates=dates,
            config=config,
            train_window_days=1095,  # 3 años fijos
            val_window_days=182,     # 6 meses (semestre de validación)
            step_days=182,
            epochs=45,
            patience=10,
            seed=seed
        )
        
        all_preds_list.append(df_preds)
        history_dict[f"seed_{seed}"] = histories
        
        seed_mean = float(df_metrics['mean_qlike'].mean())
        seed_median = float(df_metrics['median_qlike'].mean())
        seed_std = float(df_metrics['mean_qlike'].std())

        seed_results.append({
            'seed': seed,
            'mean_qlike': seed_mean,
            'median_qlike': seed_median,
            'std_folds': seed_std
        })

    # Resumen y métricas estadísticas
    summary_df = pd.DataFrame(seed_results)
    global_mean = float(summary_df['mean_qlike'].mean())
    global_std_seeds = float(summary_df['mean_qlike'].std())
    global_median = float(summary_df['median_qlike'].mean())
    global_avg_fold_std = float(summary_df['std_folds'].mean())

    print(f"\nRESUMEN FINAL: {model_name} (5 Semillas)")
    print(f" • QLIKE Promedio:          {global_mean:.4f} ± {global_std_seeds:.4f}")
    print(f" • QLIKE Mediana Promedio:  {global_median:.4f}")
    print(f" • Std Promedio entre Folds:{global_avg_fold_std:.4f}")

    # Guardado de artefactos
    summary_df.to_csv(OUTPUT_DIR / f"{model_name}_metrics_summary.csv", index=False)
    
    master_preds = pd.concat(all_preds_list, ignore_index=True)
    master_preds.to_parquet(OUTPUT_DIR / f"{model_name}_predictions_all_seeds.parquet", index=False)
    
    with open(OUTPUT_DIR / f"{model_name}_histories.json", "w") as f:
        json.dump(history_dict, f, indent=2)

if __name__ == "__main__":
    # Ejecución sistemática de las 3 arquitecturas en competencia
    execute_multi_seed_experiment("CNN_1D", VolatilidadCNN, CONFIG_CNN)
    execute_multi_seed_experiment("GRU", VolatilidadGRU, CONFIG_GRU)
    execute_multi_seed_experiment("LSTM", VolatilidadLSTM, CONFIG_LSTM)