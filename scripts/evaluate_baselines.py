import sys
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# 1. Resolución canónica de rutas
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data_with_dates

def evaluate_arch_walk_forward(
    model_name,
    returns_pct,
    df_exog,
    real_vol_proxy,
    dates,
    p=1,
    q=3,
    dist='ged',
    use_exog=False,
    train_window_days=1095,
    val_window_days=182,
    step_days=182
):
    """
    Ejecuta el Walk-Forward semestral para un modelo de la familia GARCH/EGARCH
    registrando métricas puntuales por fold y globales.
    """
    total_obs = len(returns_pct)
    eps = 1e-6
    
    fold_metrics = []
    all_predictions = []
    
    start_idx = 0
    fold_idx = 0
    
    print(f"\n--- Evaluando {model_name} (p={p}, q={q}, dist={dist.upper()}, exog={use_exog}) ---")

    while start_idx + train_window_days < total_obs:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_obs)

        if (val_end - val_start) < 90:
            break

        chunk_ret = returns_pct.iloc[:val_end]
        val_dates = dates[val_start:val_end]
        y_val_real = real_vol_proxy.iloc[val_start:val_end].values

        # Configuración de variables exógenas
        chunk_exog = df_exog.iloc[:val_end] if use_exog else None

        am = arch_model(
            chunk_ret,
            x=chunk_exog,
            mean='Constant',
            vol='EGARCH',
            p=p,
            o=1,  # Término de asimetría / apalancamiento
            q=q,
            dist=dist
        )

        try:
            last_train_date = chunk_ret.index[train_end - 1]
            res = am.fit(last_obs=last_train_date, disp='off', show_warning=False)

            # Pronóstico a 1 paso adelante alineado al target t+1
            forecasts = res.forecast(horizon=1, align='target')
            
            # Slicing riguroso: extraemos ÚNICAMENTE las predicciones del semestre out-of-sample
            test_dates = chunk_ret.index[val_start:val_end]
            sigma2_pred_pct = forecasts.variance['h.1'].loc[test_dates].values

            # Conversión de escala: de retornos porcentuales^2 a varianza decimal
            # y_val_real ya está en escala porcentual (100 * r_t)^2
            ratio = np.clip(y_val_real, eps, None) / np.clip(sigma2_pred_pct, eps, None)
            qlike_fold = ratio - np.log(ratio) - 1.0

            fold_mean = float(np.mean(qlike_fold))
            fold_median = float(np.median(qlike_fold))
            fold_std_daily = float(np.std(qlike_fold, ddof=1))

            fold_metrics.append({
                'model': model_name,
                'fold': fold_idx,
                'val_start': test_dates[0].strftime('%Y-%m-%d'),
                'val_end': test_dates[-1].strftime('%Y-%m-%d'),
                'samples': len(qlike_fold),
                'mean_qlike': fold_mean,
                'median_qlike': fold_median,
                'std_daily': fold_std_daily,
                'converged': res.convergence_flag == 0
            })

            for d, yt, sp, ql in zip(val_dates, y_val_real, sigma2_pred_pct, qlike_fold):
                all_predictions.append({
                    'Date': d,
                    'model': model_name,
                    'fold': fold_idx,
                    'y_true_pct': yt,
                    'sigma2_pred_pct': sp,
                    'qlike': ql
                })

        except Exception as e:
            print(f"Error en Fold {fold_idx}: {e}")
            fold_metrics.append({
                'model': model_name,
                'fold': fold_idx,
                'val_start': dates[val_start].strftime('%Y-%m-%d'),
                'val_end': dates[val_end - 1].strftime('%Y-%m-%d'),
                'samples': val_end - val_start,
                'mean_qlike': np.nan,
                'median_qlike': np.nan,
                'std_daily': np.nan,
                'converged': False
            })

        start_idx += step_days
        fold_idx += 1

    df_folds = pd.DataFrame(fold_metrics)
    df_preds = pd.DataFrame(all_predictions)

    # 4. Cálculo formal de Métricas Transversales y Globales
    valid_folds = df_folds.dropna(subset=['mean_qlike'])
    
    # Media de las medias de los folds (Macro-average)
    macro_mean_qlike = float(valid_folds['mean_qlike'].mean())
    # Desviación estándar ENTRE FOLDS (std transversal de estabilidad temporal)
    std_between_folds = float(valid_folds['mean_qlike'].std(ddof=1))
    # Media de las medianas de los folds
    macro_median_qlike = float(valid_folds['median_qlike'].mean())

    # Métrica Micro: sobre toda la serie temporal continua concatenada
    micro_mean_qlike = float(df_preds['qlike'].mean())
    micro_median_qlike = float(df_preds['qlike'].median())

    print(f"\nResultados Consolidados: {model_name}")
    print(f" • QLIKE Promedio (Macro ± std folds): {macro_mean_qlike:.4f} ± {std_between_folds:.4f}")
    print(f" • QLIKE Mediana Promedio (Folds):     {macro_median_qlike:.4f}")
    print(f" • QLIKE Concatenado Global (Micro):   {micro_mean_qlike:.4f} (Mediana: {micro_median_qlike:.4f})")
    print(f" • Total Folds Exitosos:               {len(valid_folds)}/{len(df_folds)}")

    return {
        'model_name': model_name,
        'macro_mean_qlike': macro_mean_qlike,
        'std_between_folds': std_between_folds,
        'macro_median_qlike': macro_median_qlike,
        'micro_mean_qlike': micro_mean_qlike,
        'micro_median_qlike': micro_median_qlike,
        'df_folds': df_folds,
        'df_preds': df_preds
    }

def main():
    data_path = ROOT / "data" / "processed" / "processed_data.csv"
    df, X, y, dates, feature_names = load_processed_data_with_dates(data_path)

    # Retorno en escala porcentual (* 100) para estabilidad de optimización MLE
    returns_pct = pd.Series(
        np.sign(df['BTC_r_oc'].values) * np.sqrt(y) * 100.0,
        index=df.index,
        name='BTC_Returns_Pct'
    )
    real_vol_proxy = returns_pct ** 2

    # Covariables exógenas seleccionadas declarativamente
    exog_cols = ['SPY_r_oc', 'SPY_r_vol', 'SPY_r_overnight', 'VIX_log']
    df_exog = df[exog_cols].copy()

    # 1. Baseline 1: EGARCH(1,3)_GED (Univariado clásico de Econometría II)
    res_egarch_ged = evaluate_arch_walk_forward(
        model_name="EGARCH(1,3)_GED",
        returns_pct=returns_pct,
        df_exog=df_exog,
        real_vol_proxy=real_vol_proxy,
        dates=df.index,
        p=1,
        q=3,
        dist='ged',
        use_exog=False,
        train_window_days=1095,
        val_window_days=182,
        step_days=182
    )

    # 2. Baseline 2: EGARCH(1,3)-X_NORMAL (Multivariado con exógenas)
    res_egarch_x = evaluate_arch_walk_forward(
        model_name="EGARCH(1,3)-X_NORMAL",
        returns_pct=returns_pct,
        df_exog=df_exog,
        real_vol_proxy=real_vol_proxy,
        dates=df.index,
        p=1,
        q=3,
        dist='normal',
        use_exog=True,
        train_window_days=1095,
        val_window_days=182,
        step_days=182
    )

    # 3. Guardado consolidado de artefactos
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    summary_rows = [
        {
            'model': res_egarch_ged['model_name'],
            'mean_qlike_macro': res_egarch_ged['macro_mean_qlike'],
            'std_folds': res_egarch_ged['std_between_folds'],
            'median_qlike_macro': res_egarch_ged['macro_median_qlike'],
            'mean_qlike_micro': res_egarch_ged['micro_mean_qlike'],
            'median_qlike_micro': res_egarch_ged['micro_median_qlike']
        },
        {
            'model': res_egarch_x['model_name'],
            'mean_qlike_macro': res_egarch_x['macro_mean_qlike'],
            'std_folds': res_egarch_x['std_between_folds'],
            'median_qlike_macro': res_egarch_x['macro_median_qlike'],
            'mean_qlike_micro': res_egarch_x['micro_mean_qlike'],
            'median_qlike_micro': res_egarch_x['micro_median_qlike']
        }
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "baselines_summary.csv", index=False)

    # Guardar desglose por fold de ambos modelos
    all_folds_df = pd.concat([res_egarch_ged['df_folds'], res_egarch_x['df_folds']], ignore_index=True)
    all_folds_df.to_csv(results_dir / "baselines_folds_detail.csv", index=False)

    # Guardar predicciones puntuales continuas para Test de Diebold-Mariano
    all_preds_df = pd.concat([res_egarch_ged['df_preds'], res_egarch_x['df_preds']], ignore_index=True)
    all_preds_df.to_parquet(results_dir / "baselines_predictions.parquet", index=False)

    print("\n" + "="*80)
    print("TABLA COMPARATIVA FINAL DE BASELINES (GUARDADA EN results/baselines_summary.csv):")
    print("="*80)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()