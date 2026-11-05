import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# 1. Configuración de rutas e importación del test DM
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.evaluation import diebold_mariano_test
except ImportError:
    # Implementación autocontenida de respaldo por robustez
    def diebold_mariano_test(loss_baseline, loss_candidate, h=1):
        d = np.asarray(loss_baseline, dtype=np.float64) - np.asarray(loss_candidate, dtype=np.float64)
        T = len(d)
        d_mean = np.mean(d)
        
        # Autocovarianzas para corrección de Newey-West (HAC)
        gamma0 = np.var(d, ddof=0)
        gamma_sum = 0.0
        for lag in range(1, h):
            weight = 1.0 - (lag / h)
            gamma_lag = np.sum((d[lag:] - d_mean) * (d[:-lag] - d_mean)) / T
            gamma_sum += 2.0 * weight * gamma_lag
            
        lr_var = gamma0 + gamma_sum
        if lr_var <= 1e-12:
            return 0.0, 1.0
            
        dm_stat = d_mean / np.sqrt(lr_var / T)
        # Factor de corrección Harvey-Leybourne-Newbold (1997)
        hln_correction = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        dm_hln = dm_stat * hln_correction
        p_val = 2.0 * stats.t.sf(np.abs(dm_hln), df=T - 1)
        return float(dm_hln), float(p_val)

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    """Calcula la pérdida QLIKE diaria de forma vectorizada y estable."""
    y = np.clip(np.asarray(y_true, dtype=np.float64), eps, None)
    s2 = np.clip(np.asarray(sigma2_pred, dtype=np.float64), eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

def get_significance_code(p_val):
    """Devuelve la convención estándar de significancia econométrica."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "·"
    return "n.s."

def load_all_models_data():
    """
    Carga y alinea las predicciones out-of-sample:
    1. Baselines unificados (Benchmark Promedio, EGARCH-GED, EGARCH-X)
    2. Modelos Deep Learning candidatos (Deep Ensemble de 5 semillas por config_id)
    3. Ensambles por arquitectura completa
    """
    print("\n[1/4] Cargando series de predicciones fuera de muestra...")
    baselines_path = RESULTS_DIR / "baselines_predictions.parquet"
    dl_path = RESULTS_DIR / "candidates_ensembles_predictions.parquet"

    if not baselines_path.exists():
        raise FileNotFoundError(f"No se encontró {baselines_path}. Ejecutá scripts/evaluate_baselines.py")
    if not dl_path.exists():
        raise FileNotFoundError(f"No se encontró {dl_path}. Ejecutá scripts/run_experiments_seeds_2.py")

    df_base = pd.read_parquet(baselines_path)
    df_base['Date'] = pd.to_datetime(df_base['Date'])

    df_dl = pd.read_parquet(dl_path)
    df_dl['Date'] = pd.to_datetime(df_dl['Date'])

    baselines = {
        'Benchmark_Promedio': df_base[df_base['model'] == 'Benchmark_Promedio'].copy(),
        'EGARCH(1,3)_GED': df_base[df_base['model'] == 'EGARCH(1,3)_GED'].copy(),
        'EGARCH(1,3)-X_NORMAL': df_base[df_base['model'] == 'EGARCH(1,3)-X_NORMAL'].copy()
    }

    dl_models = {}

    # A. Candidatos individuales (Deep Ensemble de 5 semillas por configuración)
    for cfg_id in df_dl['config_id'].unique():
        sub = df_dl[df_dl['config_id'] == cfg_id].copy()
        arch = sub['architecture'].iloc[0]
        dl_models[f"{arch}_{cfg_id}"] = pd.DataFrame({
            'Date': sub['Date'],
            'fold': sub['fold'],
            'y_true': sub['y_true'],
            'sigma2_pred': sub['sigma2_pred_mean'],
            'qlike': sub['qlike_ensemble']
        }).sort_values('Date').reset_index(drop=True)

    # B. Ensambles consolidados por familia arquitectónica
    for arch in df_dl['architecture'].unique():
        sub_arch = df_dl[df_dl['architecture'] == arch]
        ens_arch = sub_arch.groupby('Date', as_index=False).agg({
            'fold': 'first',
            'y_true': 'first',
            'sigma2_pred_mean': 'mean'
        }).rename(columns={'sigma2_pred_mean': 'sigma2_pred'})

        ens_arch['qlike'] = compute_qlike_vector(ens_arch['y_true'], ens_arch['sigma2_pred'])
        dl_models[f"{arch}_Ensemble_Arch"] = ens_arch.sort_values('Date').reset_index(drop=True)

    return baselines, dl_models

def run_comprehensive_dm_tests(baselines, dl_models):
    """
    Ejecuta la batería de pruebas de Diebold-Mariano con corrección HLN (1997)
    calculando en paralelo las pruebas bilaterales y unilaterales.
    """
    print("\n[2/4] Ejecutando contrastes estadísticos Diebold-Mariano (Bilateral y Unilateral)...")
    dm_records = []

    def evaluate_pair(base_name, cand_name, df_b, df_c, tipo_contraste):
        merged = pd.merge(
            df_b[['Date', 'qlike']].rename(columns={'qlike': 'loss_base'}),
            df_c[['Date', 'qlike']].rename(columns={'qlike': 'loss_cand'}),
            on='Date', how='inner'
        ).dropna()

        T = len(merged)
        loss_b = merged['loss_base'].values
        loss_c = merged['loss_cand'].values
        diff_mean = float(np.mean(loss_b) - np.mean(loss_c))  # Positivo = Candidato tiene MENOR pérdida

        # Ejecución del test base
        dm_stat, p_val_two = diebold_mariano_test(loss_b, loss_c, h=1)

        # Cálculo formal de la prueba unilateral (H1: E[d_t] > 0 -> Candidato supera a Base)
        # Se evalúa la cola derecha mediante la función de supervivencia (survival function)
        p_val_one = float(stats.t.sf(dm_stat, df=T - 1))

        # Indicadores de significancia
        sig_two = get_significance_code(p_val_two)
        sig_one = get_significance_code(p_val_one)

        # Veredictos textuales objetivos
        if p_val_two < 0.05:
            ver_two = f"{cand_name} supera a {base_name}" if dm_stat > 0 else f"{base_name} supera a {cand_name}"
        elif p_val_two < 0.10:
            ver_two = "Diferencia marginal (10%)"
        else:
            ver_two = "Sin diferencia significativa"

        if p_val_one < 0.05:
            ver_one = f"{cand_name} estadísticamente superior (5%)"
        elif p_val_one < 0.10:
            ver_one = f"{cand_name} superioridad marginal (10%)"
        else:
            ver_one = "No supera significativamente"

        return {
            'Tipo_Contraste': tipo_contraste,
            'Modelo_Base': base_name,
            'Modelo_Candidato': cand_name,
            'Observaciones_T': T,
            'QLIKE_Base': float(np.mean(loss_b)),
            'QLIKE_Candidato': float(np.mean(loss_c)),
            'Diferencia_Media': diff_mean,
            'DM_Stat_HLN': dm_stat,
            'p_val_bilateral': p_val_two,
            'Signif_Bilateral': sig_two,
            'p_val_unilateral': p_val_one,
            'Signif_Unilateral': sig_one,
            'Veredicto_Bilateral': ver_two,
            'Veredicto_Unilateral': ver_one
        }

    # 1. Modelos Deep Learning vs. Baselines
    for base_name, base_df in baselines.items():
        for dl_name, dl_df in dl_models.items():
            record = evaluate_pair(base_name, dl_name, base_df, dl_df, 'DL vs Baseline')
            dm_records.append(record)

    # 2. Contrastes entre los Baselines
    base_pairs = [
        ('Benchmark_Promedio', 'EGARCH(1,3)_GED'),
        ('Benchmark_Promedio', 'EGARCH(1,3)-X_NORMAL'),
        ('EGARCH(1,3)_GED', 'EGARCH(1,3)-X_NORMAL')
    ]
    for b1, b2 in base_pairs:
        record = evaluate_pair(b1, b2, baselines[b1], baselines[b2], 'Inter-Baseline')
        dm_records.append(record)

    # 3. Contrastes entre Familias de Deep Learning
    arch_pairs = [
        ('CNN_1D_Ensemble_Arch', 'LSTM_Ensemble_Arch'),
        ('CNN_1D_Ensemble_Arch', 'GRU_Ensemble_Arch'),
        ('GRU_Ensemble_Arch', 'LSTM_Ensemble_Arch')
    ]
    for a1, a2 in arch_pairs:
        record = evaluate_pair(a1, a2, dl_models[a1], dl_models[a2], 'Inter-Arquitectura DL')
        dm_records.append(record)

    df_dm = pd.DataFrame(dm_records)
    df_dm.to_csv(RESULTS_DIR / "diebold_mariano_statistical_tests.csv", index=False)
    return df_dm

def generate_comparative_plots(baselines, dl_models):
    """Genera las 4 figuras analíticas para el reporte y la defensa."""
    print("\n[3/4] Generando visualizaciones diagnósticas...")

    prom_df = baselines['Benchmark_Promedio']
    ged_df = baselines['EGARCH(1,3)_GED']
    x_df = baselines['EGARCH(1,3)-X_NORMAL']

    lstm_ens = dl_models['LSTM_Ensemble_Arch']
    cnn_ens = dl_models['CNN_1D_Ensemble_Arch']
    gru_ens = dl_models['GRU_Ensemble_Arch']

    comp = pd.merge(prom_df[['Date', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_prom', 'qlike': 'q_prom'}),
                    ged_df[['Date', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_ged', 'qlike': 'q_ged'}), on='Date')
    comp = pd.merge(comp, x_df[['Date', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_x', 'qlike': 'q_x'}), on='Date')
    comp = pd.merge(comp, lstm_ens[['Date', 'y_true', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_lstm', 'qlike': 'q_lstm'}), on='Date')
    comp = pd.merge(comp, cnn_ens[['Date', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_cnn', 'qlike': 'q_cnn'}), on='Date')
    comp = pd.merge(comp, gru_ens[['Date', 'sigma2_pred', 'qlike']].rename(columns={'sigma2_pred': 's2_gru', 'qlike': 'q_gru'}), on='Date')
    comp = comp.sort_values('Date').reset_index(drop=True)

    # FIGURA 1: Pérdida Acumulada Diferencial
    fig, ax = plt.subplots(figsize=(12, 6))
    comp['diff_prom_lstm'] = (comp['q_prom'] - comp['q_lstm']).cumsum()
    comp['diff_ged_lstm'] = (comp['q_ged'] - comp['q_lstm']).cumsum()
    comp['diff_x_lstm'] = (comp['q_x'] - comp['q_lstm']).cumsum()
    comp['diff_ged_cnn'] = (comp['q_ged'] - comp['q_cnn']).cumsum()

    ax.plot(comp['Date'], comp['diff_prom_lstm'], label='Benchmark Promedio vs. LSTM Ensamble', color='#7f7f7f', lw=1.8, ls=':')
    ax.plot(comp['Date'], comp['diff_ged_lstm'], label='EGARCH(1,3)_GED vs. LSTM Ensamble', color='#1f77b4', lw=2.2)
    ax.plot(comp['Date'], comp['diff_x_lstm'], label='EGARCH(1,3)-X vs. LSTM Ensamble', color='#2ca02c', lw=2.0)
    ax.plot(comp['Date'], comp['diff_ged_cnn'], label='EGARCH(1,3)_GED vs. CNN 1D Ensamble', color='#ff7f0e', lw=1.6, ls='--')

    ax.axhline(0, color='black', lw=1, ls='-')
    ax.set_ylabel(r'Diferencial Acumulado $\sum (QLIKE_{base} - QLIKE_{DL})$')
    ax.set_title('Evolución Temporal de la Superioridad Predictiva Fuera de Muestra\n(Pendiente Positiva = Deep Learning Supera al Baseline)')
    ax.legend(loc='upper left', frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cumulative_loss_differential.png")
    plt.close(fig)

    # FIGURA 2: Pronóstico de Varianza y Shock COVID-19
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={'height_ratios': [2, 1.2]})
    y_pct = comp['y_true'] * 10000.0
    s2_prom_pct = comp['s2_prom'] * 10000.0
    s2_ged_pct = comp['s2_ged'] * 10000.0
    s2_lstm_pct = comp['s2_lstm'] * 10000.0

    ax1.plot(comp['Date'], y_pct, color='#b0b0b0', alpha=0.5, lw=0.8, label=r'Proxy Real ($r_{t+1}^2 \times 10^4$)')
    ax1.plot(comp['Date'], s2_prom_pct, color='#7f7f7f', lw=1.2, ls=':', label='Benchmark Promedio')
    ax1.plot(comp['Date'], s2_ged_pct, color='#d62728', lw=1.3, label='EGARCH(1,3)_GED')
    ax1.plot(comp['Date'], s2_lstm_pct, color='#1f77b4', lw=1.8, label='LSTM Ensamble')
    ax1.set_ylabel(r'Varianza Porcentual $\hat{\sigma}^2_{t+1}$ (%$^2$)')
    ax1.set_title('Trayectoria de Pronóstico de Varianza de Bitcoin (Walk-Forward 2019-2025)')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', frameon=True)

    mask_covid = (comp['Date'] >= '2020-02-01') & (comp['Date'] <= '2020-06-30')
    c_covid = comp[mask_covid]
    ax2.plot(c_covid['Date'], c_covid['y_true'] * 10000.0, color='black', alpha=0.45, lw=1.0, label='Proxy Real')
    ax2.plot(c_covid['Date'], c_covid['s2_prom'] * 10000.0, color='#7f7f7f', lw=1.4, ls=':', label='Benchmark Promedio')
    ax2.plot(c_covid['Date'], c_covid['s2_ged'] * 10000.0, color='#d62728', lw=1.5, ls='--', label='EGARCH(1,3)_GED')
    ax2.plot(c_covid['Date'], c_covid['s2_lstm'] * 10000.0, color='#1f77b4', lw=2.0, label='LSTM Ensamble')
    ax2.set_ylabel(r'$\hat{\sigma}^2_{t+1}$ (%$^2$)')
    ax2.set_title('Régimen de Estrés Extremo: Ruptura de Liquidez COVID-19 (Febrero - Junio 2020)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.legend(loc='upper right', frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_variance_forecast_covid_regime.png")
    plt.close(fig)

    # FIGURA 3: Distribución Semántica del Error QLIKE (Boxplots)
    fig, ax = plt.subplots(figsize=(11, 6))
    models_data = [
        ('Benchmark Promedio', comp['q_prom'], '#7f7f7f'),
        ('EGARCH(1,3)_GED', comp['q_ged'], '#d95f02'),
        ('EGARCH(1,3)-X', comp['q_x'], '#e7298a'),
        ('CNN 1D Ensamble', comp['q_cnn'], '#ff7f0e'),
        ('GRU Ensamble', comp['q_gru'], '#2ca02c'),
        ('LSTM Ensamble', comp['q_lstm'], '#1f77b4')
    ]

    labels = [m[0] for m in models_data]
    data = [m[1] for m in models_data]
    colors = [m[2] for m in models_data]

    bplot = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showmeans=True, widths=0.45,
        meanprops={'marker': 'D', 'markerfacecolor': 'gold', 'markeredgecolor': 'black', 'markersize': 7},
        medianprops={'color': 'black', 'linewidth': 2.0},
        flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.25}
    )

    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor('black')

    for i, (_, s, col) in enumerate(models_data, start=1):
        mean_v = s.mean()
        med_v = s.median()
        q75 = s.quantile(0.75)
        y_pos = min(q75 + 0.45, 4.8)
        ax.text(
            i, y_pos, f"Media: {mean_v:.3f}\nMed: {med_v:.3f}",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=col, alpha=0.9)
        )

    ax.set_ylim(-0.2, 5.2)
    ax.set_ylabel(r'Pérdida Diaria $QLIKE_t$')
    ax.set_title('Distribución Fuera de Muestra de Pérdidas QLIKE: Baselines vs. Deep Learning\n(Evaluación Walk-Forward Bitcoin 2019–2025)')

    legend_elements = [
        Line2D([0], [0], color='black', lw=2.0, label='Línea central: Mediana'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=8, label='Rombo dorado: Media muestral')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_qlike_error_distribution_boxplots.png")
    plt.close(fig)

    # FIGURA 4: Desempeño por Fold Semestral
    folds_base = pd.read_csv(RESULTS_DIR / "baselines_folds_detail.csv")
    f_prom = folds_base[folds_base['model'] == 'Benchmark_Promedio']
    f_ged = folds_base[folds_base['model'] == 'EGARCH(1,3)_GED']

    df_dl_cands = pd.read_parquet(RESULTS_DIR / "candidates_ensembles_predictions.parquet")
    f_lstm = df_dl_cands[df_dl_cands['architecture'] == 'LSTM'].groupby('fold')['qlike_ensemble'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    n_folds = len(f_prom)
    fold_indices = np.arange(n_folds)
    w = 0.25

    ax.bar(fold_indices - w, f_prom['mean_qlike'].values[:n_folds], width=w, label='Benchmark Promedio', color='#7f7f7f', alpha=0.7)
    ax.bar(fold_indices, f_ged['mean_qlike'].values[:n_folds], width=w, label='EGARCH(1,3)_GED', color='#d62728', alpha=0.8)
    ax.bar(fold_indices + w, f_lstm.values[:n_folds], width=w, label='LSTM (Ensamble)', color='#1f77b4', alpha=0.85)

    ax.set_xticks(fold_indices)
    ax.set_xticklabels([f"Fold {f}" for f in fold_indices], rotation=45)
    ax.set_ylabel('QLIKE Medio del Fold')
    ax.set_title('Consistencia Out-of-Sample Semestre a Semestre (Walk-Forward)')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fold_by_fold_comparison.png")
    plt.close(fig)

    print(f"Figuras guardadas exitosamente en: {FIG_DIR.resolve()}")

def main():
    baselines, dl_models = load_all_models_data()
    df_dm = run_comprehensive_dm_tests(baselines, dl_models)
    generate_comparative_plots(baselines, dl_models)

    print("\n" + "="*120)
    print("BATERÍA EXHAUSTIVA DE CONTRASTES DE HIPÓTESIS: DIEBOLD-MARIANO CON CORRECCIÓN HLN (1997)")
    print("="*120)
    display_cols = [
        'Tipo_Contraste', 'Modelo_Base', 'Modelo_Candidato', 'QLIKE_Base', 'QLIKE_Candidato',
        'DM_Stat_HLN', 'p_val_bilateral', 'Signif_Bilateral', 'p_val_unilateral', 'Signif_Unilateral'
    ]
    print(df_dm[display_cols].to_string(index=False))
    print(f"\nResultados completos exportados a: {RESULTS_DIR / 'diebold_mariano_statistical_tests.csv'}")

if __name__ == "__main__":
    main()