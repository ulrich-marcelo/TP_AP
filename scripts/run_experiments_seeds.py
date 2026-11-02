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

TOP_CONFIGS = {
    "CNN_1D": {
        "model_class": VolatilidadCNN,
        "candidates": [
            {
                "config_id": "CNN_Top1",
                "timesteps": 21, "hidden_size": 24, "dropout_rate": 0.2,
                "kernel_size": 7, "num_layers": 2, "learning_rate": 0.0016989,
                "weight_decay": 0.000021872, "batch_size": 16, "run_name": "celestial-sweep-18"
            },
            {
                "config_id": "CNN_Top2",
                "timesteps": 5, "hidden_size": 24, "dropout_rate": 0.5,
                "kernel_size": 3, "num_layers": 1, "learning_rate": 0.0022024,
                "weight_decay": 0.0034483, "batch_size": 16, "run_name": "deft-sweep-10"
            },
            {
                "config_id": "CNN_Top3",
                "timesteps": 5, "hidden_size": 24, "dropout_rate": 0.5,
                "kernel_size": 3, "num_layers": 1, "learning_rate": 0.002173,
                "weight_decay": 0.0018427, "batch_size": 16, "run_name": "iconic-sweep-21"
            }
        ]
    },
    "GRU": {
        "model_class": VolatilidadGRU,
        "candidates": [
            {
                "config_id": "GRU_Top1",
                "timesteps": 21, "hidden_size": 32, "dropout_rate": 0.4,
                "num_layers": 2, "learning_rate": 0.0015367, "weight_decay": 0.000020707,
                "batch_size": 32, "run_name": "distinctive-sweep-26"
            },
            {
                "config_id": "GRU_Top2",
                "timesteps": 21, "hidden_size": 32, "dropout_rate": 0.3,
                "num_layers": 2, "learning_rate": 0.0021255, "weight_decay": 0.00010845,
                "batch_size": 32, "run_name": "zesty-sweep-16"
            },
            {
                "config_id": "GRU_Top3",
                "timesteps": 21, "hidden_size": 32, "dropout_rate": 0.4,
                "num_layers": 2, "learning_rate": 0.002296, "weight_decay": 0.003535,
                "batch_size": 32, "run_name": "morning-sweep-32"
            }
        ]
    },
    "LSTM": {
        "model_class": VolatilidadLSTM,
        "candidates": [
            {
                "config_id": "LSTM_Top1",
                "timesteps": 21, "hidden_size": 32, "dropout_rate": 0.5,
                "num_layers": 2, "learning_rate": 0.0027602, "weight_decay": 0.0002893,
                "batch_size": 16, "run_name": "copper-sweep-9"
            },
            {
                "config_id": "LSTM_Top2",
                "timesteps": 5, "hidden_size": 16, "dropout_rate": 0.2,
                "num_layers": 1, "learning_rate": 0.0013415, "weight_decay": 0.000542,
                "batch_size": 16, "run_name": "golden-sweep-34"
            },
            {
                "config_id": "LSTM_Top3",
                "timesteps": 5, "hidden_size": 32, "dropout_rate": 0.4,
                "num_layers": 1, "learning_rate": 0.0028722, "weight_decay": 0.00052284,
                "batch_size": 32, "run_name": "playful-sweep-33"
            }
        ]
    }
}

def compute_qlike_vector(y_true, sigma2_pred, eps=1e-6):
    """Calcula la pérdida QLIKE diaria vectorizada de forma numéricamente estable."""
    y = np.clip(np.asarray(y_true, dtype=np.float64), eps, None)
    s2 = np.clip(np.asarray(sigma2_pred, dtype=np.float64), eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

def run_top_n_multiseed_analysis():
    _, X, y, dates, _ = load_processed_data_with_dates(DATA_PATH)
    
    all_individual_predictions = []
    all_ensemble_predictions = []
    all_summary_rows = []
    all_histories = {}

    print("\n" + "="*85)
    print("EJECUCIÓN MULTISEMILLA: PERSISTENCIA COMPLETA DE PREDICCIONES Y ENSAMBLES")
    print("="*85)

    for arch_name, arch_data in TOP_CONFIGS.items():
        model_class = arch_data["model_class"]
        candidates = arch_data["candidates"]
        
        print(f"\n>>>> Familia Arquitectónica: {arch_name} ({len(candidates)} candidatos)")
        seed_scores = {cand["config_id"]: [] for cand in candidates}

        for cand in candidates:
            cfg_id = cand["config_id"]
            print(f"\n  -- Candidato: {cfg_id} (Sweep run: {cand.get('run_name', 'N/A')}) --")
            cand_seed_dfs = []

            for seed in SEEDS:
                print(f"     Semilla {seed:7d}...", end=" ", flush=True)
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
                
                # Enriquecimiento del registro individual
                df_preds['architecture'] = arch_name
                df_preds['config_id'] = cfg_id
                df_preds['seed'] = seed
                
                all_individual_predictions.append(df_preds)
                cand_seed_dfs.append(df_preds)
                
                all_histories[f"{arch_name}_{cfg_id}_seed_{seed}"] = histories
                
                mean_loss = float(df_preds['qlike'].mean())
                median_loss = float(df_preds['qlike'].median())
                seed_scores[cfg_id].append(mean_loss)
                print(f"QLIKE: {mean_loss:.4f} (Mediana: {median_loss:.4f})")

            # -------------------------------------------------------------
            # CONSTRUCCIÓN DEL DEEP ENSEMBLE PARA EL CANDIDATO
            # -------------------------------------------------------------
            df_cand_all = pd.concat(cand_seed_dfs, ignore_index=True)
            
            # Promedio de varianzas (no de qlikes) y cálculo de dispersión epistémica
            df_ens = df_cand_all.groupby('Date', as_index=False).agg({
                'fold': 'first',
                'y_true': 'first',
                'sigma2_pred': ['mean', 'std']
            })
            
            # Aplanar nombres de columnas tras el doble agg
            df_ens.columns = ['Date', 'fold', 'y_true', 'sigma2_pred_mean', 'sigma2_pred_std']
            df_ens['sigma2_pred_std'] = df_ens['sigma2_pred_std'].fillna(0.0)
            df_ens['architecture'] = arch_name
            df_ens['config_id'] = cfg_id
            
            # Recálculo estricto de QLIKE sobre la predicción agregada
            df_ens['qlike_ensemble'] = compute_qlike_vector(
                df_ens['y_true'].values, 
                df_ens['sigma2_pred_mean'].values
            )
            
            all_ensemble_predictions.append(df_ens)

        # Matriz y estabilidad de rankings
        df_rankings = pd.DataFrame(seed_scores, index=SEEDS)
        df_ranks_per_seed = df_rankings.rank(axis=1, ascending=True)

        for cand in candidates:
            cfg_id = cand["config_id"]
            scores = np.array(seed_scores[cfg_id])
            ranks = df_ranks_per_seed[cfg_id].values
            
            # Recuperar pérdida media del ensamble de este candidato
            ens_cand_df = next(
                e for e in all_ensemble_predictions 
                if e['config_id'].iloc[0] == cfg_id and e['architecture'].iloc[0] == arch_name
            )
            
            all_summary_rows.append({
                'architecture': arch_name,
                'config_id': cfg_id,
                'mean_seed_qlike': float(np.mean(scores)),
                'std_seeds': float(np.std(scores, ddof=1)),
                'ensemble_mean_qlike': float(ens_cand_df['qlike_ensemble'].mean()),
                'ensemble_median_qlike': float(ens_cand_df['qlike_ensemble'].median()),
                'mean_rank': float(np.mean(ranks)),
                'std_rank': float(np.std(ranks, ddof=1)),
                'timesteps': cand['timesteps'],
                'hidden_size': cand['hidden_size'],
                'dropout_rate': cand['dropout_rate'],
                'learning_rate': cand['learning_rate']
            })

    # ---------------------------------------------------------------------
    # PERSISTENCIA EN FORMATO PARQUET Y CSV
    # ---------------------------------------------------------------------
    print("\nGuardando predicciones consolidadas en 'results/'...")
    
    # 1. Nivel Individual (Semilla por Semilla)
    master_individual_df = pd.concat(all_individual_predictions, ignore_index=True)
    master_individual_df['Date'] = pd.to_datetime(master_individual_df['Date'])
    master_individual_df.to_parquet(OUTPUT_DIR / "all_seeds_predictions.parquet", index=False)
    
    # 2. Nivel Ensamble (Promediado por Candidato)
    master_ensemble_df = pd.concat(all_ensemble_predictions, ignore_index=True)
    master_ensemble_df['Date'] = pd.to_datetime(master_ensemble_df['Date'])
    master_ensemble_df.to_parquet(OUTPUT_DIR / "candidates_ensembles_predictions.parquet", index=False)
    
    # 3. Resumen de Desempeño
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "topN_multiseed_evaluation_summary.csv", index=False)
    
    with open(OUTPUT_DIR / "training_histories_all_models.json", "w") as f:
        json.dump(all_histories, f, indent=2)

    print("Artefactos exportados exitosamente:")
    print("  - results/all_seeds_predictions.parquet")
    print("  - results/candidates_ensembles_predictions.parquet")
    print("  - results/topN_multiseed_evaluation_summary.csv")

if __name__ == "__main__":
    run_top_n_multiseed_analysis()