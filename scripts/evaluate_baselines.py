import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")

# 1. Resolución canónica de rutas
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data_with_dates

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def compute_qlike(y_true, sigma2_pred, eps=1e-6):
    """Pérdida QLIKE vectorial numéricamente estable."""
    y = np.clip(np.asarray(y_true, dtype=np.float64), eps, None)
    s2 = np.clip(np.asarray(sigma2_pred, dtype=np.float64), eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

def evaluate_all_baselines(
    train_window_days=1095,
    val_window_days=182,
    step_days=182
):
    """
    Ejecuta el Walk-Forward semestral para los 3 baselines:
    1. Benchmark_Promedio: Varianza Incondicional Móvil (Rolling Unconditional Mean)
    2. EGARCH(1,3)_GED: Asimétrico Univariado con colas pesadas
    3. EGARCH(1,3)-X_NORMAL: Asimétrico Multivariado con covariables exógenas
    """
    df, X, y, dates, feature_names = load_processed_data_with_dates(DATA_PATH)
    total_obs = len(y)

    # Retorno y proxy en escala porcentual (100 * r) para convergencia numérica de MLE en arch
    returns_pct = pd.Series(
        np.sign(df['BTC_r_oc'].values) * np.sqrt(y) * 100.0,
        index=df.index,
        name='BTC_Returns_Pct'
    )
    real_vol_proxy_pct = returns_pct ** 2  # Escala %^2 (10000 * y)

    # Covariables exógenas para EGARCH-X
    exog_cols = ['SPY_r_oc', 'SPY_r_vol', 'SPY_r_overnight', 'VIX_log']
    df_exog = df[exog_cols].copy()

    all_predictions = []
    all_folds_detail = []

    print("\n" + "="*85)
    print("EJECUTANDO EVALUACIÓN UNIFICADA DE BASELINES (WALK-FORWARD 2019-2025)")
    print("="*85)
    print(f"Total datos: {total_obs} | Ventana entrenamiento: {train_window_days} | Test: {val_window_days}")

    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_obs:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_obs)

        if (val_end - val_start) < 90:
            break

        val_dates = df.index[val_start:val_end]
        n_samples = val_end - val_start

        # Datos en escala decimal (estándar Deep Learning)
        y_train_dec = y[train_start:train_end]
        y_val_dec = y[val_start:val_end]

        # Datos en escala porcentual (para arch_model)
        chunk_ret = returns_pct.iloc[:val_end]
        chunk_exog = df_exog.iloc[:val_end]
        test_dates = chunk_ret.index[val_start:val_end]
        last_train_date = chunk_ret.index[train_end - 1]

        print(f"\n>> Procesando Fold {fold_idx:02d} ({val_dates[0].strftime('%Y-%m-%d')} a {val_dates[-1].strftime('%Y-%m-%d')}) | {n_samples} días")

        # -------------------------------------------------------------
        # 1. BASELINE TRIVIAL: PROMEDIO MÓVIL (VARIANZA INCONDICIONAL)
        # -------------------------------------------------------------
        rolling_var_dec = float(np.mean(y_train_dec))
        pred_promedio_dec = np.full(n_samples, rolling_var_dec)
        qlike_promedio = compute_qlike(y_val_dec, pred_promedio_dec)

        all_folds_detail.append({
            'model': 'Benchmark_Promedio',
            'fold': fold_idx,
            'val_start': val_dates[0].strftime('%Y-%m-%d'),
            'val_end': val_dates[-1].strftime('%Y-%m-%d'),
            'samples': n_samples,
            'mean_qlike': float(np.mean(qlike_promedio)),
            'median_qlike': float(np.median(qlike_promedio)),
            'std_daily': float(np.std(qlike_promedio, ddof=1)),
            'converged': True
        })

        for d, yt, sp, ql in zip(val_dates, y_val_dec, pred_promedio_dec, qlike_promedio):
            all_predictions.append({
                'Date': pd.to_datetime(d),
                'fold': fold_idx,
                'model': 'Benchmark_Promedio',
                'y_true': yt,
                'sigma2_pred': sp,
                'sigma2_pred_pct': sp * 10000.0,
                'qlike': ql
            })

        # -------------------------------------------------------------
        # 2. BASELINE ECONOMÉTRICO 1: EGARCH(1,3)_GED (Univariado)
        # -------------------------------------------------------------
        am_ged = arch_model(
            chunk_ret,
            mean='Constant',
            vol='EGARCH',
            p=1, o=1, q=3,
            dist='ged'
        )
        try:
            res_ged = am_ged.fit(last_obs=last_train_date, disp='off', show_warning=False)
            f_ged = res_ged.forecast(horizon=1, align='target')
            s2_ged_pct = f_ged.variance['h.1'].loc[test_dates].values
            s2_ged_dec = s2_ged_pct / 10000.0
            qlike_ged = compute_qlike(y_val_dec, s2_ged_dec)
            conv_ged = (res_ged.convergence_flag == 0)
        except Exception as e:
            print(f"   [!] Error en EGARCH(1,3)_GED Fold {fold_idx}: {e}")
            s2_ged_dec = pred_promedio_dec
            s2_ged_pct = pred_promedio_dec * 10000.0
            qlike_ged = qlike_promedio
            conv_ged = False

        all_folds_detail.append({
            'model': 'EGARCH(1,3)_GED',
            'fold': fold_idx,
            'val_start': val_dates[0].strftime('%Y-%m-%d'),
            'val_end': val_dates[-1].strftime('%Y-%m-%d'),
            'samples': n_samples,
            'mean_qlike': float(np.mean(qlike_ged)),
            'median_qlike': float(np.median(qlike_ged)),
            'std_daily': float(np.std(qlike_ged, ddof=1)),
            'converged': conv_ged
        })

        for d, yt, sp, spp, ql in zip(val_dates, y_val_dec, s2_ged_dec, s2_ged_pct, qlike_ged):
            all_predictions.append({
                'Date': pd.to_datetime(d),
                'fold': fold_idx,
                'model': 'EGARCH(1,3)_GED',
                'y_true': yt,
                'sigma2_pred': sp,
                'sigma2_pred_pct': spp,
                'qlike': ql
            })

        # -------------------------------------------------------------
        # 3. BASELINE ECONOMÉTRICO 2: EGARCH(1,3)-X_NORMAL (Multivariado)
        # -------------------------------------------------------------
        am_x = arch_model(
            chunk_ret,
            x=chunk_exog,
            mean='Constant',
            vol='EGARCH',
            p=1, o=1, q=3,
            dist='normal'
        )
        try:
            res_x = am_x.fit(last_obs=last_train_date, disp='off', show_warning=False)
            f_x = res_x.forecast(horizon=1, align='target')
            s2_x_pct = f_x.variance['h.1'].loc[test_dates].values
            s2_x_dec = s2_x_pct / 10000.0
            qlike_x = compute_qlike(y_val_dec, s2_x_dec)
            conv_x = (res_x.convergence_flag == 0)
        except Exception as e:
            print(f"   [!] Error en EGARCH(1,3)-X_NORMAL Fold {fold_idx}: {e}")
            s2_x_dec = pred_promedio_dec
            s2_x_pct = pred_promedio_dec * 10000.0
            qlike_x = qlike_promedio
            conv_x = False

        all_folds_detail.append({
            'model': 'EGARCH(1,3)-X_NORMAL',
            'fold': fold_idx,
            'val_start': val_dates[0].strftime('%Y-%m-%d'),
            'val_end': val_dates[-1].strftime('%Y-%m-%d'),
            'samples': n_samples,
            'mean_qlike': float(np.mean(qlike_x)),
            'median_qlike': float(np.median(qlike_x)),
            'std_daily': float(np.std(qlike_x, ddof=1)),
            'converged': conv_x
        })

        for d, yt, sp, spp, ql in zip(val_dates, y_val_dec, s2_x_dec, s2_x_pct, qlike_x):
            all_predictions.append({
                'Date': pd.to_datetime(d),
                'fold': fold_idx,
                'model': 'EGARCH(1,3)-X_NORMAL',
                'y_true': yt,
                'sigma2_pred': sp,
                'sigma2_pred_pct': spp,
                'qlike': ql
            })

        print(f"   • Promedio:  QLIKE = {np.mean(qlike_promedio):.4f}")
        print(f"   • EGARCH:    QLIKE = {np.mean(qlike_ged):.4f}")
        print(f"   • EGARCH-X:  QLIKE = {np.mean(qlike_x):.4f}")

        start_idx += step_days
        fold_idx += 1

    # Consolidación final en DataFrames
    df_preds_all = pd.DataFrame(all_predictions)
    df_folds_all = pd.DataFrame(all_folds_detail)

    # Resumen macro y micro
    summary_rows = []
    for model_name in ['Benchmark_Promedio', 'EGARCH(1,3)_GED', 'EGARCH(1,3)-X_NORMAL']:
        sub_folds = df_folds_all[df_folds_all['model'] == model_name]
        sub_preds = df_preds_all[df_preds_all['model'] == model_name]

        summary_rows.append({
            'model': model_name,
            'mean_qlike_macro': float(sub_folds['mean_qlike'].mean()),
            'std_folds': float(sub_folds['mean_qlike'].std(ddof=1)),
            'median_qlike_macro': float(sub_folds['median_qlike'].mean()),
            'mean_qlike_micro': float(sub_preds['qlike'].mean()),
            'median_qlike_micro': float(sub_preds['qlike'].median()),
            'total_observations': len(sub_preds)
        })

    df_summary = pd.DataFrame(summary_rows)

    # Persistencia de artefactos
    df_preds_all.to_parquet(RESULTS_DIR / "baselines_predictions.parquet", index=False)
    df_folds_all.to_csv(RESULTS_DIR / "baselines_folds_detail.csv", index=False)
    df_summary.to_csv(RESULTS_DIR / "baselines_summary.csv", index=False)

    print("\n" + "="*85)
    print("RESUMEN GENERAL DE BASELINES (GUARDADO EN results/baselines_summary.csv):")
    print("="*85)
    print(df_summary.to_string(index=False))
    print(f"\nPredicciones exportadas a: {RESULTS_DIR / 'baselines_predictions.parquet'}")

if __name__ == "__main__":
    evaluate_all_baselines()