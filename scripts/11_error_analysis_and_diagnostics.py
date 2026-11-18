import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import warnings

warnings.filterwarnings("ignore")

# 1. Configuración de Rutas
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) not in sys.path else Path(".")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Jerarquía canónica de modelos para visualización y tablas
MODEL_ORDER = [
    'Benchmark_Promedio', 
    'EGARCH(1,3)_GED', 
    'EGARCH(1,3)-X_NORMAL', 
    'CNN_1D_Ensemble', 
    'GRU_Ensemble', 
    'LSTM_Ensemble'
]

REGIME_ORDER = [
    '1. Muestra Completa (2019-2025)',
    '2. Shock COVID-19 (Feb-Jun 2020)',
    '3. Bull Market Post-Halving (Oct 2020-May 2021)',
    '4. Criptoinvierno / Bear (Nov 2021-Dic 2022)',
    '5. Madurez / ETF Spot (2023-2025)',
    '6. Baja Volatilidad (y <= Q50)',
    '7. Moderada Volatilidad (Q50 < y <= Q90)',
    '8. Estrés Extremo / Colas (y > Q90)'
]

try:
    from src.evaluation import diebold_mariano_test
except ImportError:
    def diebold_mariano_test(loss_baseline, loss_candidate, h=1):
        d = np.asarray(loss_baseline, dtype=np.float64) - np.asarray(loss_candidate, dtype=np.float64)
        T = len(d)
        mean_d = np.mean(d)
        gamma0 = np.var(d, ddof=0)
        gamma_sum = 0.0
        for lag in range(1, h):
            weight = 1.0 - (lag / h)
            gamma_lag = np.sum((d[lag:] - mean_d) * (d[:-lag] - mean_d)) / T
            gamma_sum += 2.0 * weight * gamma_lag
        lr_var = gamma0 + gamma_sum
        if lr_var <= 1e-12:
            return 0.0, 1.0
        dm_stat = (mean_d / np.sqrt(lr_var / T)) * np.sqrt((T + 1 - 2*h + (h*(h-1))/T) / T)
        p_val = 2.0 * stats.t.sf(np.abs(dm_stat), df=T - 1)
        return float(dm_stat), float(p_val)

def compute_qlike_vector(y_true, sigma2_pred, eps=1e-6):
    y = np.clip(np.asarray(y_true, dtype=np.float64), eps, None)
    s2 = np.clip(np.asarray(sigma2_pred, dtype=np.float64), eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

def load_data():
    p_ens = RESULTS_DIR / "candidates_ensembles_predictions.parquet"
    p_base = RESULTS_DIR / "baselines_predictions.parquet"

    if not p_ens.exists() or not p_base.exists():
        raise FileNotFoundError("Faltan archivos Parquet en results/. Ejecutá previamente el pipeline.")

    df_ens = pd.read_parquet(p_ens)
    df_ens['Date'] = pd.to_datetime(df_ens['Date'])

    df_base = pd.read_parquet(p_base)
    df_base['Date'] = pd.to_datetime(df_base['Date'])

    df_raw = None
    if DATA_PATH.exists():
        df_raw = pd.read_csv(DATA_PATH)
        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        else:
            df_raw.index = pd.to_datetime(df_raw.index)
            df_raw = df_raw.reset_index().rename(columns={'index': 'Date'})

    return df_ens, df_base, df_raw

def build_model_predictions_dict(df_ens, df_base):
    models = {}
    for m in ['Benchmark_Promedio', 'EGARCH(1,3)_GED', 'EGARCH(1,3)-X_NORMAL']:
        sub = df_base[df_base['model'] == m].sort_values('Date').reset_index(drop=True)
        models[m] = pd.DataFrame({
            'Date': sub['Date'],
            'y_true': sub['y_true'],
            'sigma2_pred': sub['sigma2_pred'],
            'qlike': sub['qlike']
        })

    for arch in ['CNN_1D', 'GRU', 'LSTM']:
        sub_arch = df_ens[df_ens['architecture'] == arch]
        ens = sub_arch.groupby('Date', as_index=False).agg({
            'y_true': 'first',
            'sigma2_pred_mean': 'mean'
        }).rename(columns={'sigma2_pred_mean': 'sigma2_pred'})
        ens['qlike'] = compute_qlike_vector(ens['y_true'].values, ens['sigma2_pred'].values)
        models[f"{arch}_Ensemble"] = ens.sort_values('Date').reset_index(drop=True)

    return models

# ==============================================================================
# 1. DISECCIÓN POR REGÍMENES DE MERCADO
# ==============================================================================
def run_market_regimes_dissection(models_dict):
    print("\n" + "="*95)
    print("DISECCIÓN 1: ANÁLISIS DE PÉRDIDAS POR REGÍMENES DE MERCADO Y ESTRÉS")
    print("="*95)

    ref_df = models_dict['EGARCH(1,3)_GED'].copy()
    y_full = ref_df['y_true'].values
    dates_full = ref_df['Date']

    q50 = float(np.quantile(y_full, 0.50))
    q90 = float(np.quantile(y_full, 0.90))

    regimes_defs = {
        '1. Muestra Completa (2019-2025)': dates_full >= dates_full.min(),
        '2. Shock COVID-19 (Feb-Jun 2020)': (dates_full >= '2020-02-01') & (dates_full <= '2020-06-30'),
        '3. Bull Market Post-Halving (Oct 2020-May 2021)': (dates_full >= '2020-10-01') & (dates_full <= '2021-05-31'),
        '4. Criptoinvierno / Bear (Nov 2021-Dic 2022)': (dates_full >= '2021-11-01') & (dates_full <= '2022-12-31'),
        '5. Madurez / ETF Spot (2023-2025)': dates_full >= '2023-01-01',
        '6. Baja Volatilidad (y <= Q50)': y_full <= q50,
        '7. Moderada Volatilidad (Q50 < y <= Q90)': (y_full > q50) & (y_full <= q90),
        '8. Estrés Extremo / Colas (y > Q90)': y_full > q90
    }

    records = []
    days_dict = {}

    for reg_name in REGIME_ORDER:
        mask = regimes_defs[reg_name]
        n_days = int(np.sum(mask))
        days_dict[reg_name] = n_days
        if n_days == 0:
            continue

        egarch_ged_loss = models_dict['EGARCH(1,3)_GED'].loc[mask, 'qlike'].values

        for m_name in MODEL_ORDER:
            df_m = models_dict[m_name]
            sub = df_m.loc[mask]
            q_vals = sub['qlike'].values
            
            mean_q = float(np.mean(q_vals))
            med_q = float(np.median(q_vals))
            std_q = float(np.std(q_vals, ddof=1)) if len(q_vals) > 1 else 0.0
            q84_q = float(np.quantile(q_vals, 0.84))
            max_q = float(np.max(q_vals))

            p_val_dm = np.nan
            dm_stat = np.nan
            if 'COVID-19' in reg_name or 'Completa' in reg_name or 'Madurez' in reg_name:
                if m_name != 'EGARCH(1,3)_GED':
                    dm_stat, p_val_dm = diebold_mariano_test(egarch_ged_loss, q_vals, h=1)

            diff_vs_egarch = float(np.mean(egarch_ged_loss) - mean_q)

            records.append({
                'Régimen': reg_name,
                'Días': n_days,
                'Modelo': m_name,
                'QLIKE_Media': mean_q,
                'QLIKE_Mediana': med_q,
                'QLIKE_Std': std_q,
                'QLIKE_Q84': q84_q,
                'QLIKE_Max': max_q,
                'Diff_vs_EGARCH': diff_vs_egarch,
                'DM_Stat_HLN': dm_stat,
                'p_val_DM': p_val_dm
            })

    # A. Tabla larga ordenada categóricamente
    df_regimes = pd.DataFrame(records)
    df_regimes['Régimen'] = pd.Categorical(df_regimes['Régimen'], categories=REGIME_ORDER, ordered=True)
    df_regimes['Modelo'] = pd.Categorical(df_regimes['Modelo'], categories=MODEL_ORDER, ordered=True)
    df_regimes = df_regimes.sort_values(['Régimen', 'Modelo']).reset_index(drop=True)
    df_regimes.to_csv(OUTPUT_DIR / "error_analysis_regimes.csv", index=False)

    # B. Matriz comparativa horizontal para contraste inmediato entre arquitecturas
    pivot_media = df_regimes.pivot(index='Régimen', columns='Modelo', values='QLIKE_Media')
    pivot_diff = df_regimes.pivot(index='Régimen', columns='Modelo', values='Diff_vs_EGARCH')

    matrix_rows = []
    for reg_name in REGIME_ORDER:
        row = {'Régimen': reg_name, 'Días': days_dict[reg_name]}
        for m in MODEL_ORDER:
            row[f"{m}_QLIKE"] = pivot_media.loc[reg_name, m]
            row[f"{m}_Diff"] = pivot_diff.loc[reg_name, m]
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(OUTPUT_DIR / "error_analysis_regimes_matrix.csv", index=False)

    print("\n--- MATRIZ HORIZONTAL DE COMPARACIÓN: QLIKE MEDIO POR RÉGIMEN ---")
    display_matrix = df_matrix[['Régimen', 'Días'] + [f"{m}_QLIKE" for m in MODEL_ORDER]]
    print(display_matrix.to_string(index=False))

    return df_regimes, df_matrix

# ==============================================================================
# 2. DISECCIÓN DE RESIDUOS ESTANDARIZADOS Y ADECUACIÓN ECONOMÉTRICA
# ==============================================================================
def run_residuals_diagnostics_dissection(models_dict, df_raw):
    print("\n" + "="*95)
    print("DISECCIÓN 2: DIAGNÓSTICO ECONOMÉTRICO DE RESIDUOS ESTANDARIZADOS")
    print("="*95)

    records = []
    signs_dict = {}
    if df_raw is not None and 'BTC_r_oc' in df_raw.columns and 'Date' in df_raw.columns:
        sub_raw = df_raw[['Date', 'BTC_r_oc']].drop_duplicates('Date').set_index('Date')
        signs_dict = np.sign(sub_raw['BTC_r_oc']).to_dict()

    for m_name in MODEL_ORDER:
        df_m = models_dict[m_name]
        y = np.clip(df_m['y_true'].values, 1e-8, None)
        s2 = np.clip(df_m['sigma2_pred'].values, 1e-8, None)

        z2 = y / s2
        dates = df_m['Date'].values
        signs = np.array([signs_dict.get(pd.to_datetime(d), 1.0) for d in dates])
        z = np.sqrt(z2) if np.all(signs == 1.0) else (signs * np.sqrt(z2))
        T = len(z2)

        mean_z2 = float(np.mean(z2))
        bias_pct = float((mean_z2 - 1.0) * 100.0)
        median_z2 = float(np.median(z2))
        std_z2 = float(np.std(z2, ddof=1))
        kurt_z2 = float(stats.kurtosis(z2, fisher=True))
        kurt_raw_y = float(stats.kurtosis(y, fisher=True))
        kurt_reduction_pct = float((1.0 - (kurt_z2 / kurt_raw_y)) * 100.0)

        lb_res = acorr_ljungbox(z2, lags=[5, 10, 20], return_df=True)
        lb_p5 = float(lb_res.loc[5, 'lb_pvalue'])
        lb_p10 = float(lb_res.loc[10, 'lb_pvalue'])
        lb_p20 = float(lb_res.loc[20, 'lb_pvalue'])

        try:
            arch_lm_5 = het_arch(z, nlags=5)
            arch_p5 = float(arch_lm_5[1])
            arch_lm_10 = het_arch(z, nlags=10)
            arch_p10 = float(arch_lm_10[1])
        except Exception:
            arch_p5, arch_p10 = np.nan, np.nan

        passed_filter = "Aprobado (Efectos ARCH eliminados)" if lb_p10 > 0.05 else "Rechazado (Memoria remanente)"

        records.append({
            'Modelo': m_name,
            'Observaciones_T': T,
            'E[z^2]_Sesgo': mean_z2,
            'Sesgo_%': bias_pct,
            'Mediana_z^2': median_z2,
            'Std_z^2': std_z2,
            'Curtosis_Proxy_Bruto': kurt_raw_y,
            'Exceso_Curtosis_z^2': kurt_z2,
            'Reduccion_Curtosis_%': kurt_reduction_pct,
            'LjungBox_p_Lag5': lb_p5,
            'LjungBox_p_Lag10': lb_p10,
            'LjungBox_p_Lag20': lb_p20,
            'ARCH_LM_p_Lag5': arch_p5,
            'ARCH_LM_p_Lag10': arch_p10,
            'Filtro_Heterocedasticidad': passed_filter
        })

    df_residuals = pd.DataFrame(records)
    df_residuals['Modelo'] = pd.Categorical(df_residuals['Modelo'], categories=MODEL_ORDER, ordered=True)
    df_residuals = df_residuals.sort_values('Modelo').reset_index(drop=True)
    df_residuals.to_csv(OUTPUT_DIR / "error_analysis_residuals_diagnostics.csv", index=False)

    print("\n--- DIAGNÓSTICO DE RESIDUOS ESTANDARIZADOS: SESGO Y REDUCCIÓN DE CURTOSIS ---")
    disp_cols = [
        'Modelo', 'E[z^2]_Sesgo', 'Sesgo_%', 'Exceso_Curtosis_z^2', 
        'Reduccion_Curtosis_%', 'LjungBox_p_Lag10', 'Filtro_Heterocedasticidad'
    ]
    print(df_residuals[disp_cols].to_string(index=False))

    return df_residuals

def main():
    df_ens, df_base, df_raw = load_data()
    models_dict = build_model_predictions_dict(df_ens, df_base)
    run_market_regimes_dissection(models_dict)
    run_residuals_diagnostics_dissection(models_dict, df_raw)
    print(f"\nTablas actualizadas y persistidas en '{OUTPUT_DIR.resolve()}'.")

if __name__ == "__main__":
    main()