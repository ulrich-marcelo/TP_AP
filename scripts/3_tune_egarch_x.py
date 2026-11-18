import sys
from pathlib import Path
import itertools
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

def run_egarch_x_walk_forward(returns, df_exog, real_vol_proxy, p, q, dist, window_size=1095, semester_len=182):
    """
    Ejecuta el Walk-Forward semestral idéntico al baseline.ipynb para una especificación dada
    y retorna la serie out-of-sample de QLIKE y sus métricas.
    """
    total_obs = len(returns)
    pred_vars_chunks = []
    failed_folds = 0
    total_folds = 0

    for i in range(0, total_obs - window_size, semester_len):
        current_train_end = window_size + i
        current_test_end = min(current_train_end + semester_len, total_obs)

        if current_train_end >= total_obs:
            break

        total_folds += 1
        chunk_ret = returns.iloc[:current_test_end]
        chunk_exog = df_exog.iloc[:current_test_end]

        am = arch_model(
            chunk_ret,
            x=chunk_exog,
            mean='Constant',
            vol='EGARCH',
            p=p,
            o=1,  # Término de asimetría / efecto apalancamiento
            q=q,
            dist=dist
        )

        try:
            last_train_date = chunk_ret.index[current_train_end]
            res = am.fit(last_obs=last_train_date, disp='off', show_warning=False)

            if res.convergence_flag != 0:
                failed_folds += 1

            # Pronóstico fuera de muestra alineado al target t+1
            forecasts = res.forecast(horizon=1, align='target')
            
            # Extraemos la varianza proyectada correspondiente únicamente a la ventana out-of-sample
            test_dates = chunk_ret.index[current_train_end:current_test_end]
            sem_forecast = forecasts.variance['h.1'].loc[test_dates]
            pred_vars_chunks.append(sem_forecast)

        except Exception:
            failed_folds += 1
            continue

    if len(pred_vars_chunks) == 0:
        return None

    # Concatenación y alineación temporal de predicciones
    v_pred = pd.concat(pred_vars_chunks).dropna()
    common_dates = v_pred.index.intersection(real_vol_proxy.index)

    if len(common_dates) == 0:
        return None

    v_pred = v_pred.loc[common_dates]
    v_real = real_vol_proxy.loc[common_dates].squeeze()

    # Cálculo puntual de la pérdida QLIKE (escala-invariante)
    # y / sigma^2 - ln(y / sigma^2) - 1
    qlike_series = (v_real / v_pred) - np.log(v_real / v_pred) - 1.0

    return {
        'mean_qlike': float(qlike_series.mean()),
        'median_qlike': float(qlike_series.median()),
        'std_qlike': float(qlike_series.std()),
        'failed_folds': failed_folds,
        'total_folds': total_folds,
        'qlike_series': qlike_series,
        'v_pred': v_pred
    }

def find_optimal_egarch_x_by_qlike(initial_train_days=1095, semester_len=182):
    """
    Grid Search exhaustivo de EGARCH-X evaluado rigurosamente mediante QLIKE Out-of-Sample.
    """
    data_path = ROOT / "data" / "processed" / "processed_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

    df, X, y, dates, feature_names = load_processed_data_with_dates(data_path)

    # Reconstrucción de retornos en escala porcentual (100 * r_t) para estabilidad de arch_model
    # target_r2_t1 es (r_{t+1})^2 en escala decimal
    returns_pct = pd.Series(
        np.sign(df['BTC_r_oc'].values) * np.sqrt(y) * 100.0,
        index=df.index,
        name='BTC_Returns_Pct'
    )

    # Proxy de varianza real para la evaluación de QLIKE
    real_vol_proxy = returns_pct.iloc[initial_train_days:] ** 2

    # Variables exógenas seleccionadas declarativamente por nombre
    exog_cols = ['SPY_r_oc', 'SPY_r_vol', 'SPY_r_overnight', 'VIX_log']
    df_exog = df[exog_cols].copy()

    # Grilla de hiperparámetros
    p_orders = [1, 2]
    q_orders = [1, 2, 3]
    distributions = ['ged', 't', 'normal']

    grid_results = []

    print("\n" + "="*80)
    print("INICIANDO GRID SEARCH DE EGARCH-X BASADO EN QLIKE OUT-OF-SAMPLE (WALK-FORWARD)")
    print("="*80)
    print(f"Observaciones totales: {len(df)} | Ventana inicial: {initial_train_days} días | Bloque test: {semester_len} días")
    print(f"Variables exógenas incorporadas: {exog_cols}\n")

    for p, q, dist in itertools.product(p_orders, q_orders, distributions):
        spec_name = f"EGARCH({p},{q})-X_{dist.upper()}"
        print(f"Evaluando {spec_name}...", end=" ", flush=True)

        res_dict = run_egarch_x_walk_forward(
            returns=returns_pct,
            df_exog=df_exog,
            real_vol_proxy=real_vol_proxy,
            p=p,
            q=q,
            dist=dist,
            window_size=initial_train_days,
            semester_len=semester_len
        )

        if res_dict is not None and not np.isnan(res_dict['mean_qlike']):
            print(f"-> Mean QLIKE: {res_dict['mean_qlike']:.4f} | Median: {res_dict['median_qlike']:.4f} | Fails: {res_dict['failed_folds']}/{res_dict['total_folds']}")
            grid_results.append({
                'spec': spec_name,
                'p': p,
                'q': q,
                'dist': dist,
                'mean_qlike': res_dict['mean_qlike'],
                'median_qlike': res_dict['median_qlike'],
                'std_qlike': res_dict['std_qlike'],
                'failed_folds': res_dict['failed_folds'],
                'total_folds': res_dict['total_folds']
            })
        else:
            print("-> Falló convergencia.")

    results_df = pd.DataFrame(grid_results)
    
    # Ordenamos estrictamente por QLIKE Promedio fuera de muestra
    results_df = results_df.sort_values(by='mean_qlike').reset_index(drop=True)

    print("\n" + "="*80)
    print("RANKING FINAL DE MODELOS EGARCH-X SEGÚN ERROR QLIKE (OUT-OF-SAMPLE):")
    print("="*80)
    print(results_df[['spec', 'mean_qlike', 'median_qlike', 'std_qlike', 'failed_folds']].to_string(index=False))

    best_model = results_df.iloc[0]
    print("\n" + "-"*80)
    print(f"ESPECIFICACIÓN ÓPTIMA SELECCIONADA POR QLIKE: {best_model['spec']}")
    print(f"Mean QLIKE: {best_model['mean_qlike']:.4f} | Median QLIKE: {best_model['median_qlike']:.4f}")
    print("-"*80)

    # Persistencia del reporte
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(results_dir / "egarch_x_qlike_grid_search.csv", index=False)
    print(f"\nResultados guardados en: {results_dir / 'egarch_x_qlike_grid_search.csv'}")

    return best_model

if __name__ == "__main__":
    find_optimal_egarch_x_by_qlike()