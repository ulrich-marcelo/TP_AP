import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Configuración de rutas e importación del test DM
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import diebold_mariano_test

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Configuración estética de Matplotlib para reporte académico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300
})

def compute_qlike_vector(y_true, sigma2_pred, eps=1e-6):
    """Calcula la pérdida QLIKE diaria vectorizada de forma numéricamente estable."""
    y = np.clip(y_true, eps, None)
    s2 = np.clip(sigma2_pred, eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

def load_and_prepare_data():
    """
    Carga y alinea las predicciones de baselines y los mejores candidatos de DL.
    Calcula además la predicción del ensamble de 5 semillas para cada arquitectura.
    """
    print("\n[1/4] Cargando predicciones fuera de muestra...")

    # Cargar predicciones de baselines econométricos
    df_base = pd.read_parquet(RESULTS_DIR / "baselines_predictions.parquet")
    df_base['Date'] = pd.to_datetime(df_base['Date'])

    egarch_ged = df_base[df_base['model'] == "EGARCH(1,3)_GED"].copy()
    egarch_x = df_base[df_base['model'] == "EGARCH(1,3)-X_NORMAL"].copy()

    # Modelos campeones seleccionados a partir del análisis multiesemilla
    best_configs = {
        'LSTM': {'file': 'LSTM_topN_predictions.parquet', 'config_id': 'LSTM_Top1'},
        'CNN_1D': {'file': 'CNN_1D_topN_predictions.parquet', 'config_id': 'CNN_Top2'},
        'GRU': {'file': 'GRU_topN_predictions.parquet', 'config_id': 'GRU_Top2'}
    }

    dl_dfs = {}

    for arch, info in best_configs.items():
        parquet_path = RESULTS_DIR / info['file']
        if not parquet_path.exists():
            raise FileNotFoundError(f"No se encontró {parquet_path}. Ejecutá antes run_experiments_seeds.py.")

        df_arch = pd.read_parquet(parquet_path)
        df_arch['Date'] = pd.to_datetime(df_arch['Date'])
        df_top = df_arch[df_arch['config_id'] == info['config_id']].copy()

        # 1. Ensamble de las 5 semillas: promedio de sigma2_pred por día
        df_ens = df_top.groupby('Date', as_index=False).agg({
            'y_true': 'first',
            'sigma2_pred': 'mean',
            'fold': 'first'
        })
        df_ens['qlike'] = compute_qlike_vector(df_ens['y_true'].values, df_ens['sigma2_pred'].values)
        dl_dfs[f"{arch}_Ensemble"] = df_ens

        # 2. Representante de semilla única (Seed 42 por consistencia con sweeps)
        df_seed42 = df_top[df_top['seed'] == 42].copy()
        if len(df_seed42) > 0:
            dl_dfs[f"{arch}_Seed42"] = df_seed42

    return egarch_ged, egarch_x, dl_dfs

def run_diebold_mariano_battery(egarch_ged, egarch_x, dl_dfs):
    """
    Ejecuta el test de Diebold-Mariano con corrección de Harvey-Leybourne-Newbold
    haciendo inner merge estricto sobre las fechas comunes.
    """
    print("\n[2/4] Ejecutando batería del Test de Diebold-Mariano (HLN 1997)...")
    dm_results = []

    baselines = {
        'EGARCH(1,3)_GED': egarch_ged,
        'EGARCH(1,3)-X_NORMAL': egarch_x
    }

    for base_name, base_df in baselines.items():
        for dl_name, dl_df in dl_dfs.items():
            # Alineación temporal biunívoca
            merged = pd.merge(
                base_df[['Date', 'qlike']].rename(columns={'qlike': 'loss_base'}),
                dl_df[['Date', 'qlike']].rename(columns={'qlike': 'loss_dl'}),
                on='Date',
                how='inner'
            ).dropna()

            T = len(merged)
            dm_stat, p_val = diebold_mariano_test(
                loss_baseline=merged['loss_base'].values,
                loss_candidate=merged['loss_dl'].values,
                h=1
            )

            mean_base = merged['loss_base'].mean()
            mean_dl = merged['loss_dl'].mean()
            diff_mean = mean_base - mean_dl  # Positivo = DL supera al baseline

            signif = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            conclusion = "DL estadísticamente superior" if (p_val < 0.05 and dm_stat > 0) else "Sin diferencia significativa"

            dm_results.append({
                'Baseline': base_name,
                'Modelo_DL': dl_name,
                'Observaciones_T': T,
                'QLIKE_Baseline': mean_base,
                'QLIKE_DL': mean_dl,
                'Diferencia_Media': diff_mean,
                'DM_Stat_HLN': dm_stat,
                'p_value': p_val,
                'Significancia': signif,
                'Veredicto': conclusion
            })

    # Test complementario: EGARCH vs EGARCH-X
    merged_base = pd.merge(
        egarch_ged[['Date', 'qlike']].rename(columns={'qlike': 'loss_ged'}),
        egarch_x[['Date', 'qlike']].rename(columns={'qlike': 'loss_x'}),
        on='Date', how='inner'
    ).dropna()
    dm_stat, p_val = diebold_mariano_test(merged_base['loss_ged'], merged_base['loss_x'], h=1)
    dm_results.append({
        'Baseline': 'EGARCH(1,3)_GED',
        'Modelo_DL': 'EGARCH(1,3)-X_NORMAL',
        'Observaciones_T': len(merged_base),
        'QLIKE_Baseline': merged_base['loss_ged'].mean(),
        'QLIKE_DL': merged_base['loss_x'].mean(),
        'Diferencia_Media': merged_base['loss_ged'].mean() - merged_base['loss_x'].mean(),
        'DM_Stat_HLN': dm_stat,
        'p_value': p_val,
        'Significancia': "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")),
        'Veredicto': "Exógenas aportan significativamente" if (p_val < 0.05 and dm_stat > 0) else "Sin diferencia"
    })

    # Test complementario: LSTM_Ensemble vs CNN_1D_Ensemble
    merged_dl = pd.merge(
        dl_dfs['LSTM_Ensemble'][['Date', 'qlike']].rename(columns={'qlike': 'loss_lstm'}),
        dl_dfs['CNN_1D_Ensemble'][['Date', 'qlike']].rename(columns={'qlike': 'loss_cnn'}),
        on='Date', how='inner'
    ).dropna()
    dm_stat, p_val = diebold_mariano_test(merged_dl['loss_cnn'], merged_dl['loss_lstm'], h=1)
    dm_results.append({
        'Baseline': 'CNN_1D_Ensemble',
        'Modelo_DL': 'LSTM_Ensemble',
        'Observaciones_T': len(merged_dl),
        'QLIKE_Baseline': merged_dl['loss_cnn'].mean(),
        'QLIKE_DL': merged_dl['loss_lstm'].mean(),
        'Diferencia_Media': merged_dl['loss_cnn'].mean() - merged_dl['loss_lstm'].mean(),
        'DM_Stat_HLN': dm_stat,
        'p_value': p_val,
        'Significancia': "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")),
        'Veredicto': "LSTM superior a CNN" if (p_val < 0.05 and dm_stat > 0) else "Equivalentes estadísticos"
    })

    df_dm = pd.DataFrame(dm_results)
    df_dm.to_csv(RESULTS_DIR / "diebold_mariano_statistical_tests.csv", index=False)
    return df_dm

def generate_comparative_plots(egarch_ged, egarch_x, dl_dfs):
    """Genera las 4 figuras clave para el trabajo final."""
    print("\n[3/4] Generando visualizaciones diagnósticas...")

    lstm_ens = dl_dfs['LSTM_Ensemble']
    cnn_ens = dl_dfs['CNN_1D_Ensemble']

    # Merge para sincronización temporal completa
    comp = pd.merge(
        egarch_ged[['Date', 'y_true_pct', 'sigma2_pred_pct', 'qlike']].rename(
            columns={'y_true_pct': 'y_true_base', 'sigma2_pred_pct': 's2_egarch', 'qlike': 'qlike_egarch'}
        ),
        egarch_x[['Date', 'sigma2_pred_pct', 'qlike']].rename(
            columns={'sigma2_pred_pct': 's2_egarch_x', 'qlike': 'qlike_egarch_x'}
        ),
        on='Date'
    )
    comp = pd.merge(
        comp,
        lstm_ens[['Date', 'y_true', 'sigma2_pred', 'qlike']].rename(
            columns={'y_true': 'y_true_dl', 'sigma2_pred': 's2_lstm', 'qlike': 'qlike_lstm'}
        ),
        on='Date'
    )
    comp = pd.merge(
        comp,
        cnn_ens[['Date', 'sigma2_pred', 'qlike']].rename(
            columns={'sigma2_pred': 's2_cnn', 'qlike': 'qlike_cnn'}
        ),
        on='Date'
    ).sort_values('Date').reset_index(drop=True)

    # Re-escalar varianza decimal a porcentual para graficar en la misma escala
    # y_true_dl está en escala decimal: (retorno)^2. Multiplicar por 10000 para pasar a %^2
    comp['s2_lstm_pct'] = comp['s2_lstm'] * 10000.0
    comp['s2_cnn_pct'] = comp['s2_cnn'] * 10000.0
    comp['y_true_pct'] = comp['y_true_dl'] * 10000.0

    # -------------------------------------------------------------
    # FIGURA 1: Pérdida Acumulada Diferencial (Cumulative Loss Differential)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    comp['cum_diff_ged_lstm'] = (comp['qlike_egarch'] - comp['qlike_lstm']).cumsum()
    comp['cum_diff_x_lstm'] = (comp['qlike_egarch_x'] - comp['qlike_lstm']).cumsum()
    comp['cum_diff_ged_cnn'] = (comp['qlike_egarch'] - comp['qlike_cnn']).cumsum()

    ax.plot(comp['Date'], comp['cum_diff_ged_lstm'], label='EGARCH(1,3)_GED vs. LSTM (Ensemble)', color='#1f77b4', lw=2.0)
    ax.plot(comp['Date'], comp['cum_diff_x_lstm'], label='EGARCH(1,3)-X vs. LSTM (Ensemble)', color='#2ca02c', lw=2.0)
    ax.plot(comp['Date'], comp['cum_diff_ged_cnn'], label='EGARCH(1,3)_GED vs. CNN 1D (Ensemble)', color='#ff7f0e', lw=1.6, ls='--')

    ax.axhline(0, color='black', lw=1, ls=':')
    ax.set_ylabel(r'Diferencial Acumulado $\sum (QLIKE_{base} - QLIKE_{DL})$')
    ax.set_title('Evolución Temporal de la Superioridad Predictiva (Pendiente Positiva = DL Supera al Baseline)')
    ax.legend(loc='upper left', frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cumulative_loss_differential.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 2: Series de Varianza Condicional Pronosticada y Zoom COVID-19
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1.2]})

    # Serie completa fuera de muestra
    ax1.plot(comp['Date'], comp['y_true_pct'], color='#b0b0b0', alpha=0.6, lw=0.8, label=r'Proxy Real ($r_{t+1}^2 \times 10^4$)')
    ax1.plot(comp['Date'], comp['s2_egarch'], color='#d62728', lw=1.2, label='EGARCH(1,3)_GED')
    ax1.plot(comp['Date'], comp['s2_egarch_x'], color='#9467bd', lw=1.2, label='EGARCH(1,3)-X')
    ax1.plot(comp['Date'], comp['s2_lstm_pct'], color='#1f77b4', lw=1.8, label='LSTM (Ensemble)')
    ax1.set_ylabel(r'Varianza Porcentual $\hat{\sigma}^2_{t+1}$')
    ax1.set_title('Predicción de Varianza Diaria de Bitcoin (Out-of-Sample Walk-Forward 2019-2025)')
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_yscale('log')

    # Zoom en régimen de alta tensión: Crash de Marzo 2020 (COVID-19)
    covid_mask = (comp['Date'] >= '2020-02-01') & (comp['Date'] <= '2020-06-30')
    comp_covid = comp[covid_mask]

    ax2.plot(comp_covid['Date'], comp_covid['y_true_pct'], color='black', alpha=0.5, lw=1.0, label='Proxy Real')
    ax2.plot(comp_covid['Date'], comp_covid['s2_egarch'], color='#d62728', lw=1.5, ls='--', label='EGARCH(1,3)_GED')
    ax2.plot(comp_covid['Date'], comp_covid['s2_lstm_pct'], color='#1f77b4', lw=2.0, label='LSTM (Ensemble)')
    ax2.set_ylabel(r'$\hat{\sigma}^2_{t+1}$ (%$^2$)')
    ax2.set_title('Detalle de Régimen Crítico: Shock de Liquidez COVID-19 (Febrero - Junio 2020)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.legend(loc='upper right', frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_variance_forecast_covid_regime.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 3: Distribución del Error QLIKE (Boxplots y Colas)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = [
        comp['qlike_egarch'],
        comp['qlike_egarch_x'],
        comp['qlike_lstm'],
        comp['qlike_cnn']
    ]
    labels = ['EGARCH_GED', 'EGARCH-X', 'LSTM_Ensemble', 'CNN_Ensemble']

    bplot = ax.boxplot(plot_data, label=labels, patch_artist=True, showmeans=True,
                       meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'black', 'markersize': 6},
                       flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})

    colors = ['#ff9999', '#c2c2f0', '#99ff99', '#ffcc99']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylim(-0.5, 6.0)
    ax.set_ylabel('Pérdida QLIKE Diaria')
    ax.set_title('Distribución de la Función de Pérdida QLIKE (Puntos Rojos = Medias Globales)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_qlike_error_distribution_boxplots.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # FIGURA 4: Desempeño por Fold Semestral (Estabilidad Temporal)
    # -------------------------------------------------------------
    # Cargar detalles de folds
    folds_base = pd.read_csv(RESULTS_DIR / "baselines_folds_detail.csv")
    folds_base = folds_base[folds_base['model'] == "EGARCH(1,3)_GED"]

    fig, ax = plt.subplots(figsize=(12, 5))
    # Calcular media de DL por fold
    fold_means_lstm = lstm_ens.groupby('fold')['qlike'].mean()
    fold_indices = np.arange(len(folds_base))
    width = 0.35

    ax.bar(fold_indices - width/2, folds_base['mean_qlike'], width=width, label='EGARCH(1,3)_GED', color='#d62728', alpha=0.8)
    ax.bar(fold_indices + width/2, fold_means_lstm.values[:len(folds_base)], width=width, label='LSTM (Ensemble)', color='#1f77b4', alpha=0.8)

    ax.set_xticks(fold_indices)
    ax.set_xticklabels([f"Fold {f}" for f in fold_indices], rotation=45)
    ax.set_ylabel('QLIKE Medio del Fold')
    ax.set_title('Consistencia Out-of-Sample Semestre a Semestre (Walk-Forward)')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fold_by_fold_comparison.png")
    plt.close(fig)

    print(f"Figuras guardadas exitosamente en: {FIG_DIR}")

def main():
    egarch_ged, egarch_x, dl_dfs = load_and_prepare_data()
    df_dm = run_diebold_mariano_battery(egarch_ged, egarch_x, dl_dfs)
    generate_comparative_plots(egarch_ged, egarch_x, dl_dfs)

    print("\n" + "="*95)
    print("BATERÍA DE TESTS DE DIEBOLD-MARIANO CON CORRECCIÓN HLN (1997):")
    print("="*95)
    cols_show = ['Baseline', 'Modelo_DL', 'Observaciones_T', 'QLIKE_Baseline', 'QLIKE_DL', 'DM_Stat_HLN', 'p_value', 'Significancia']
    print(df_dm[cols_show].to_string(index=False))
    print("\nResultados completos exportados a: results/diebold_mariano_statistical_tests.csv")

if __name__ == "__main__":
    main()