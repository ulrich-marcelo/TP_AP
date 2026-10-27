import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow

# 1. Resolución canónica de rutas
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data_with_dates

def run_trivial_benchmark(
    train_window_days=1095,  # 3 años fijos
    val_window_days=182,     # 6 meses (1 semestre de test)
    step_days=182
):
    """
    Evalúa los Benchmarks Triviales de Varianza Incondicional 
    (Móvil y Estática) sobre el esquema de Walk-Forward.
    """
    data_path = ROOT / "data" / "processed" / "processed_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

    df, X, y, dates, _ = load_processed_data_with_dates(data_path)
    total_days = len(y)
    eps = 1e-6

    # 1. Varianza incondicional estática inicial (primeros 1095 días)
    static_uncond_var = float(np.mean(y[:train_window_days]))

    fold_metrics = []
    all_predictions = []

    start_idx = 0
    fold_idx = 0

    print("\n" + "="*80)
    print("EVALUACIÓN DEL BENCHMARK TRIVIAL (VARIANZA INCONDICIONAL) - WALK-FORWARD")
    print("="*80)
    print(f"Total observaciones: {total_days} | Ventana entrenamiento: {train_window_days} días")
    print(f"Varianza Incondicional Estática Inicial (2016-2018): {static_uncond_var:.6f}\n")

    while start_idx + train_window_days < total_days:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_days)

        if (val_end - val_start) < 90:
            break

        y_train = y[train_start:train_end]
        y_val = y[val_start:val_end]
        val_dates = df.index[val_start:val_end]

        # Variante A: Varianza incondicional local móvil
        rolling_uncond_var = float(np.mean(y_train))
        pred_rolling = np.full_like(y_val, rolling_uncond_var)

        # Variante B: Varianza incondicional estática inicial fija
        pred_static = np.full_like(y_val, static_uncond_var)

        # Cálculo de QLIKE para Variante A (Móvil)
        r_roll = np.clip(y_val, eps, None) / np.clip(pred_rolling, eps, None)
        qlike_rolling = r_roll - np.log(r_roll) - 1.0

        # Cálculo de QLIKE para Variante B (Estática)
        r_stat = np.clip(y_val, eps, None) / np.clip(pred_static, eps, None)
        qlike_static = r_stat - np.log(r_stat) - 1.0

        fold_metrics.append({
            'fold': fold_idx,
            'val_start': df.index[val_start].strftime('%Y-%m-%d'),
            'val_end': df.index[val_end - 1].strftime('%Y-%m-%d'),
            'samples': len(y_val),
            'var_rolling': rolling_uncond_var,
            'mean_qlike_rolling': float(np.mean(qlike_rolling)),
            'median_qlike_rolling': float(np.median(qlike_rolling)),
            'std_qlike_rolling': float(np.std(qlike_rolling)),
            'mean_qlike_static': float(np.mean(qlike_static)),
            'median_qlike_static': float(np.median(qlike_static)),
        })

        for d, yt, pr, ps, qr, qs in zip(val_dates, y_val, pred_rolling, pred_static, qlike_rolling, qlike_static):
            all_predictions.append({
                'Date': d,
                'fold': fold_idx,
                'y_true': yt,
                'sigma2_pred_rolling': pr,
                'sigma2_pred_static': ps,
                'qlike_rolling': qr,
                'qlike_static': qs
            })

        start_idx += step_days
        fold_idx += 1

    df_folds = pd.DataFrame(fold_metrics)
    df_preds = pd.DataFrame(all_predictions)

    # Métricas consolidadas
    mean_roll = float(df_preds['qlike_rolling'].mean())
    median_roll = float(df_preds['qlike_rolling'].median())
    std_roll_folds = float(df_folds['mean_qlike_rolling'].std())

    mean_stat = float(df_preds['qlike_static'].mean())
    median_stat = float(df_preds['qlike_static'].median())
    std_stat_folds = float(df_folds['mean_qlike_static'].std())

    print("Desglose por Fold Semestral:")
    print(df_folds[['fold', 'val_start', 'val_end', 'var_rolling', 'mean_qlike_rolling', 'median_qlike_rolling']].to_string(index=False))

    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO DEL BENCHMARK TRIVIAL (OUT-OF-SAMPLE 2019-2025):")
    print("="*80)
    print(f"1. Varianza Incondicional Móvil (Rolling):")
    print(f"   • QLIKE Promedio:          {mean_roll:.4f} ± {std_roll_folds:.4f}")
    print(f"   • QLIKE Mediana:           {median_roll:.4f}")
    print(f"2. Varianza Incondicional Estática (Static Initial):")
    print(f"   • QLIKE Promedio:          {mean_stat:.4f} ± {std_stat_folds:.4f}")
    print(f"   • QLIKE Mediana:           {median_stat:.4f}")
    print("="*80)

    # Guardado de artefactos
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    df_folds.to_csv(results_dir / "trivial_benchmark_folds.csv", index=False)
    df_preds.to_parquet(results_dir / "trivial_benchmark_predictions.parquet", index=False)
    print(f"\nArtefactos guardados exitosamente en:\n - {results_dir / 'trivial_benchmark_folds.csv'}\n - {results_dir / 'trivial_benchmark_predictions.parquet'}")

    return df_folds, df_preds

if __name__ == "__main__":
    run_trivial_benchmark()