import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    VolatilidadCNN,
    VolatilidadGRU,
    VolatilidadLSTM,
    load_processed_data_with_dates,
    run_walk_forward
)

SEEDS = [42, 123, 456, 789, 7122001]
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =====================================================================
# TOP-N CANDIDATOS POR ARQUITECTURA (Derivados de los Sweeps en W&B)
# =====================================================================
TOP_CONFIGS = {
    "CNN_1D": {
        "model_class": VolatilidadCNN,
        "candidates": [
            {
                "config_id": "CNN_Top1",
                "timesteps": 7, "hidden_size": 24, "dropout_rate": 0.3,
                "kernel_size": 3, "num_layers": 2, "learning_rate": 1e-3,
                "weight_decay": 1e-4, "batch_size": 32
            },
            {
                "config_id": "CNN_Top2",
                "timesteps": 5, "hidden_size": 32, "dropout_rate": 0.4,
                "kernel_size": 3, "num_layers": 1, "learning_rate": 8e-4,
                "weight_decay": 5e-4, "batch_size": 32
            },
            {
                "config_id": "CNN_Top3",
                "timesteps": 14, "hidden_size": 24, "dropout_rate": 0.3,
                "kernel_size": 5, "num_layers": 2, "learning_rate": 1e-3,
                "weight_decay": 1e-3, "batch_size": 64
            }
        ]
    },
    "GRU": {
        "model_class": VolatilidadGRU,
        "candidates": [
            {
                "config_id": "GRU_Top1",
                "timesteps": 7, "hidden_size": 32, "dropout_rate": 0.3,
                "num_layers": 1, "learning_rate": 1e-3, "weight_decay": 1e-4,
                "batch_size": 32
            },
            {
                "config_id": "GRU_Top2",
                "timesteps": 5, "hidden_size": 24, "dropout_rate": 0.3,
                "num_layers": 1, "learning_rate": 1.2e-3, "weight_decay": 1e-4,
                "batch_size": 32
            },
            {
                "config_id": "GRU_Top3",
                "timesteps": 14, "hidden_size": 32, "dropout_rate": 0.4,
                "num_layers": 2, "learning_rate": 8e-4, "weight_decay": 5e-4,
                "batch_size": 64
            }
        ]
    },
    "LSTM": {
        "model_class": VolatilidadLSTM,
        "candidates": [
            {
                "config_id": "LSTM_Top1",
                "timesteps": 7, "hidden_size": 32, "dropout_rate": 0.3,
                "num_layers": 1, "learning_rate": 1e-3, "weight_decay": 1e-4,
                "batch_size": 32
            },
            {
                "config_id": "LSTM_Top2",
                "timesteps": 5, "hidden_size": 32, "dropout_rate": 0.4,
                "num_layers": 1, "learning_rate": 8e-4, "weight_decay": 1e-4,
                "batch_size": 32
            },
            {
                "config_id": "LSTM_Top3",
                "timesteps": 14, "hidden_size": 24, "dropout_rate": 0.3,
                "num_layers": 2, "learning_rate": 6e-4, "weight_decay": 1e-3,
                "batch_size": 64
            }
        ]
    }
}

def run_top_n_multiseed_analysis():
    _, X, y, dates, _ = load_processed_data_with_dates(DATA_PATH)
    
    all_summary_rows = []
    
    print("\n" + "="*85)
    print("INICIANDO EVALUACIÓN MULTISEMILLA DEL TOP-N Y ESTABILIDAD DE RANKING")
    print("="*85)

    for arch_name, arch_data in TOP_CONFIGS.items():
        model_class = arch_data["model_class"]
        candidates = arch_data["candidates"]
        
        print(f"\n>>>> Familia Arquitectónica: {arch_name} ({len(candidates)} candidatos)")
        
        # Matriz para rastrear el QLIKE medio por (Semilla x Configuración)
        seed_scores = {cand["config_id"]: [] for cand in candidates}
        arch_preds = []
        arch_histories = {}

        for cand in candidates:
            cfg_id = cand["config_id"]
            print(f"\n  -- Evaluando Configuración: {cfg_id} en 5 Semillas --")
            
            for seed in SEEDS:
                print(f"     Semilla {seed}...", end=" ", flush=True)
                df_metrics, df_preds, histories = run_walk_forward(
                    model_class=model_class,
                    X_raw=X,
                    y_raw=y,
                    dates=dates,
                    config=cand,
                    train_window_days=1095,
                    val_window_days=182,
                    step_days=182,
                    epochs=45,
                    patience=10,
                    seed=seed
                )
                
                seed_mean_qlike = float(df_metrics['mean_qlike'].mean())
                seed_median_qlike = float(df_metrics['median_qlike'].mean())
                seed_std_folds = float(df_metrics['mean_qlike'].std())

                seed_scores[cfg_id].append(seed_mean_qlike)
                
                df_preds['architecture'] = arch_name
                df_preds['config_id'] = cfg_id
                arch_preds.append(df_preds)
                
                arch_histories[f"{cfg_id}_seed_{seed}"] = histories
                print(f"QLIKE: {seed_mean_qlike:.4f} (Mediana: {seed_median_qlike:.4f})")

        # Construcción de DataFrame de posiciones y ranking
        df_rankings = pd.DataFrame(seed_scores, index=SEEDS)
        # Ranking en cada semilla (1 = mejor / menor pérdida)
        df_ranks_per_seed = df_rankings.rank(axis=1, ascending=True)

        print(f"\nMatriz de QLIKEs Medios ({arch_name}):")
        print(df_rankings.to_string())
        
        print(f"\nMatriz de Posiciones / Ranking por Semilla ({arch_name}):")
        print(df_ranks_per_seed.to_string())

        # Análisis estadístico de estabilidad del ranking
        for cand in candidates:
            cfg_id = cand["config_id"]
            scores = np.array(seed_scores[cfg_id])
            ranks = df_ranks_per_seed[cfg_id].values
            
            all_summary_rows.append({
                'architecture': arch_name,
                'config_id': cfg_id,
                'mean_qlike': float(np.mean(scores)),
                'std_seeds': float(np.std(scores, ddof=1)),
                'min_qlike': float(np.min(scores)),
                'max_qlike': float(np.max(scores)),
                'mean_rank': float(np.mean(ranks)),
                'std_rank': float(np.std(ranks, ddof=1)),
                'timesteps': cand['timesteps'],
                'hidden_size': cand['hidden_size'],
                'dropout_rate': cand['dropout_rate'],
                'learning_rate': cand['learning_rate']
            })

        # Guardado de artefactos por arquitectura
        pd.concat(arch_preds, ignore_index=True).to_parquet(
            OUTPUT_DIR / f"{arch_name}_topN_predictions.parquet", index=False
        )
        df_ranks_per_seed.to_csv(OUTPUT_DIR / f"{arch_name}_ranking_stability.csv")
        with open(OUTPUT_DIR / f"{arch_name}_histories.json", "w") as f:
            json.dump(arch_histories, f, indent=2)

    # Reporte Ejecutivo Final
    df_global_summary = pd.DataFrame(all_summary_rows)
    df_global_summary = df_global_summary.sort_values(
        by=['architecture', 'mean_qlike']
    ).reset_index(drop=True)
    
    df_global_summary.to_csv(OUTPUT_DIR / "topN_multiseed_evaluation_summary.csv", index=False)

    print("\n" + "="*85)
    print("REPORTE FINAL CONSOLIDADO (TOP-N EVALUADO EN 5 SEMILLAS):")
    print("="*85)
    print(df_global_summary[[
        'architecture', 'config_id', 'mean_qlike', 'std_seeds', 'mean_rank', 'std_rank'
    ]].to_string(index=False))
    print("\nArtefactos guardados exitosamente en la carpeta 'results/'.")

if __name__ == "__main__":
    run_top_n_multiseed_analysis()