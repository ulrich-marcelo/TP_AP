import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data_with_dates, load_returns_reference
from src.evaluation import (
    compute_qlike_vector,
    diebold_mariano_full,
    get_significance_code,
)

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
})


def _detect_egarch_x_name(df_base: pd.DataFrame) -> str:
    spec_path = RESULTS_DIR / "egarch_x_best_spec.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("spec") in df_base["model"].unique():
            return cfg["spec"]

    others = [m for m in df_base["model"].unique() if m not in ("Benchmark_Promedio", "EGARCH(1,3)_GED")]
    if others:
        return str(others[0])
    raise ValueError("No se encontró modelo EGARCH-X en baselines_predictions.parquet.")


def build_unified_oos_panel():
    """
    Construye una tabla maestra diaria alineada (2019-2026) combinando:
      - Proxy de varianza realizada y_true = r_{t+1}^2 y retorno con signo r_{t+1}
      - Variables de estado observadas en t (VIX_log, BTC_r_oc, SPY_r_oc)
      - Pronósticos sigma2_pred y pérdidas QLIKE de todos los baselines y ensambles DL
    """
    baselines_path = RESULTS_DIR / "baselines_predictions.parquet"
    arch_ens_path = RESULTS_DIR / "architecture_ensembles_predictions.parquet"

    for p in (baselines_path, arch_ens_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}. Ejecutá previamente los scripts 4 y 8.")

    df_base = pd.read_parquet(baselines_path)
    df_base["Date"] = pd.to_datetime(df_base["Date"])

    df_arch = pd.read_parquet(arch_ens_path)
    df_arch["Date"] = pd.to_datetime(df_arch["Date"])

    egarch_x_name = _detect_egarch_x_name(df_base)

    # Cargar features y retornos con signo para reconstruir r_{t+1} exacto
    df_proc, _, _, _, _ = load_processed_data_with_dates(DATA_PATH, phase="all")
    df_ref = load_returns_reference(DATA_PATH, phase="all")

    df_ctx = pd.DataFrame(index=df_proc.index)
    next_dates = pd.Series(df_proc.index, index=df_proc.index).shift(-1)
    fallback_dates = pd.Series(df_proc.index + pd.Timedelta(days=1), index=df_proc.index)
    df_ctx["Event_Date_t1"] = next_dates.fillna(fallback_dates)

    r_cc_t1 = df_ref["BTC_r_cc"].shift(-1)
    fallback_sign = np.sign(df_proc["BTC_r_oc"].shift(-1)).fillna(1.0)
    r_cc_t1 = r_cc_t1.fillna(fallback_sign * np.sqrt(df_proc["target_r2_t1"]))

    df_ctx["r_t1"] = r_cc_t1.values
    df_ctx["VIX_level_t"] = np.exp(df_proc["VIX_log"].values)
    df_ctx["SPY_r_oc_t"] = df_proc["SPY_r_oc"].values
    df_ctx = df_ctx.reset_index().rename(columns={"Date": "Date"})
    df_ctx["Date"] = pd.to_datetime(df_ctx["Date"])

    # Tabla base diaria desde Benchmark_Promedio
    panel = (
        df_base[df_base["model"] == "Benchmark_Promedio"][["Date", "phase", "fold", "y_true", "sigma2_pred", "qlike"]]
        .rename(columns={"sigma2_pred": "s2_Benchmark_Promedio", "qlike": "q_Benchmark_Promedio"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    panel = pd.merge(panel, df_ctx, on="Date", how="left")

    # Agregar EGARCH univariado y EGARCH-X
    for m_name in ["EGARCH(1,3)_GED", egarch_x_name]:
        sub = df_base[df_base["model"] == m_name][["Date", "sigma2_pred", "qlike"]].rename(
            columns={"sigma2_pred": f"s2_{m_name}", "qlike": f"q_{m_name}"}
        )
        panel = pd.merge(panel, sub, on="Date", how="inner")

    # Agregar ensambles Hybrid_2x2 de las 3 arquitecturas DL
    for arch in ["CNN_1D", "GRU", "LSTM"]:
        sub = df_arch[(df_arch["architecture"] == arch) & (df_arch["ensemble_type"] == "Hybrid_2x2")][
            ["Date", "sigma2_pred", "qlike"]
        ].rename(columns={"sigma2_pred": f"s2_{arch}_Hybrid_2x2", "qlike": f"q_{arch}_Hybrid_2x2"})
        panel = pd.merge(panel, sub, on="Date", how="inner")

    # Agregar Super-Ensamble de las 3 arquitecturas (60 modelos)
    panel["s2_DL_SuperEnsemble_60M"] = (
        panel["s2_CNN_1D_Hybrid_2x2"] + panel["s2_GRU_Hybrid_2x2"] + panel["s2_LSTM_Hybrid_2x2"]
    ) / 3.0
    panel["q_DL_SuperEnsemble_60M"] = compute_qlike_vector(
        panel["y_true"].values, panel["s2_DL_SuperEnsemble_60M"].values
    )

    model_list = [
        "Benchmark_Promedio",
        "EGARCH(1,3)_GED",
        egarch_x_name,
        "CNN_1D_Hybrid_2x2",
        "GRU_Hybrid_2x2",
        "LSTM_Hybrid_2x2",
        "DL_SuperEnsemble_60M",
    ]
    return panel, model_list, egarch_x_name


# ==============================================================================
# BLOQUE 1: DISECCIÓN POR REGÍMENES DE MERCADO Y TEST DM CONDICIONAL
# ==============================================================================
def analyze_regimes_and_conditional_dm(panel: pd.DataFrame, model_list: list, egarch_x_name: str) -> pd.DataFrame:
    """
    Evalúa el desempeño de cada modelo por régimen de volatilidad (definidos
    estrictamente con cuantiles de Fase A), régimen macro (VIX) y asimetría (Crash vs Rally),
    incluyendo contrastes Diebold-Mariano condicionales dentro de cada régimen.
    """
    phase_a_mask = panel["phase"] == "Fase_A_Seleccion"
    y_a = panel.loc[phase_a_mask, "y_true"].values
    vix_a = panel.loc[phase_a_mask, "VIX_level_t"].values

    p50_y = float(np.percentile(y_a, 50))
    p90_y = float(np.percentile(y_a, 90))
    p98_y = float(np.percentile(y_a, 98))
    p85_vix = float(np.percentile(vix_a, 85))

    regimes = [
        ("1_Calma (< P50)", panel["y_true"] < p50_y),
        ("2_Normal (P50 - P90)", (panel["y_true"] >= p50_y) & (panel["y_true"] < p90_y)),
        ("3_Turbulencia_Alta (P90 - P98)", (panel["y_true"] >= p90_y) & (panel["y_true"] < p98_y)),
        ("4_Estres_Extremo (> P98)", panel["y_true"] >= p98_y),
        ("5_Crash_Severo (r < 0 & y >= P90)", (panel["y_true"] >= p90_y) & (panel["r_t1"] < 0)),
        ("6_Rally_Explosivo (r >= 0 & y >= P90)", (panel["y_true"] >= p90_y) & (panel["r_t1"] >= 0)),
        ("7_Contagio_Macro (VIX >= P85)", panel["VIX_level_t"] >= p85_vix),
        ("8_Test_Ciego_2026 (Fase B)", panel["phase"] == "Fase_B_Test_Ciego"),
    ]

    rows = []
    for reg_name, mask in regimes:
        sub = panel[mask].reset_index(drop=True)
        n_obs = len(sub)
        if n_obs < 10:
            continue

        q_ged = sub["q_EGARCH(1,3)_GED"].values
        q_x = sub[f"q_{egarch_x_name}"].values

        for m in model_list:
            q_m = sub[f"q_{m}"].values
            s2_m = sub[f"s2_{m}"].values

            win_vs_ged = float(np.mean(q_m < q_ged) * 100.0) if m != "EGARCH(1,3)_GED" else 50.0
            win_vs_x = float(np.mean(q_m < q_x) * 100.0) if m != egarch_x_name else 50.0

            # Test DM condicional vs EGARCH(1,3)_GED y vs EGARCH-X
            if m != "EGARCH(1,3)_GED":
                dm_ged = diebold_mariano_full(q_ged, q_m, h=1)
                dm_stat_ged = dm_ged["dm_stat"]
                p_uni_ged = dm_ged["p_val_unilateral"]
                sig_ged = get_significance_code(p_uni_ged)
            else:
                dm_stat_ged, p_uni_ged, sig_ged = 0.0, 0.5, "-"

            if m != egarch_x_name:
                dm_x = diebold_mariano_full(q_x, q_m, h=1)
                dm_stat_x = dm_x["dm_stat"]
                p_uni_x = dm_x["p_val_unilateral"]
                sig_x = get_significance_code(p_uni_x)
            else:
                dm_stat_x, p_uni_x, sig_x = 0.0, 0.5, "-"

            rows.append({
                "Regimen": reg_name,
                "Dias": int(n_obs),
                "Modelo": m,
                "Mean_QLIKE": float(np.mean(q_m)),
                "Median_QLIKE": float(np.median(q_m)),
                "Std_QLIKE": float(np.std(q_m, ddof=1)),
                "Max_QLIKE_PeorDia": float(np.max(q_m)),
                "Mean_Var_Pred_pct2": float(np.mean(s2_m) * 10000.0),
                "WinRate_vs_GED_pct": win_vs_ged,
                "WinRate_vs_EGARCHX_pct": win_vs_x,
                "DM_vs_GED": dm_stat_ged,
                "p_uni_vs_GED": p_uni_ged,
                "Sig_vs_GED": sig_ged,
                "DM_vs_EGARCHX": dm_stat_x,
                "p_uni_vs_EGARCHX": p_uni_x,
                "Sig_vs_EGARCHX": sig_x,
            })

    df_regimes = pd.DataFrame(rows)
    df_regimes.to_csv(RESULTS_DIR / "regime_error_analysis.csv", index=False)
    return df_regimes


# ==============================================================================
# BLOQUE 2: AUDITORÍA CASO A CASO DE LOS TOP-15 SHOCKS EXTREMOS (2019–2026)
# ==============================================================================
def analyze_top_extreme_shocks(panel: pd.DataFrame, model_list: list, top_n: int = 15) -> pd.DataFrame:
    """
    Compara el pronóstico de varianza y la pérdida QLIKE día por día en los
    top_n mayores shocks de volatilidad ocurridos en el período out-of-sample.
    """
    top_df = panel.nlargest(top_n, "y_true").copy().reset_index(drop=True)
    rows = []

    for rank_idx, (_, r) in enumerate(top_df.iterrows(), start=1):
        rec = {
            "Rank": rank_idx,
            "Feature_Date_t": r["Date"].strftime("%Y-%m-%d"),
            "Event_Date_t1": pd.to_datetime(r["Event_Date_t1"]).strftime("%Y-%m-%d"),
            "Fase": r["phase"],
            "BTC_Ret_t1_pct": float(r["r_t1"] * 100.0),
            "Realized_r2_pct2": float(r["y_true"] * 10000.0),
            "VIX_t": float(r["VIX_level_t"]),
        }
        for m in model_list:
            rec[f"PredVar_{m}_pct2"] = float(r[f"s2_{m}"] * 10000.0)
            rec[f"QLIKE_{m}"] = float(r[f"q_{m}"])
        rows.append(rec)

    df_shocks = pd.DataFrame(rows)
    df_shocks.to_csv(RESULTS_DIR / "top15_extreme_shocks_models.csv", index=False)
    return df_shocks


# ==============================================================================
# BLOQUE 3: DIAGNÓSTICO DE RESIDUOS ESTANDARIZADOS z_{t+1} = r_{t+1} / sigma_{t+1}
# ==============================================================================
def analyze_standardized_residuals(panel: pd.DataFrame, model_list: list) -> pd.DataFrame:
    """
    Evalúa la calibración econométrica de los residuos estandarizados z = r_{t+1} / hat_sigma_{t+1}:
      - Varianza unitaria (E[z^2] ≈ 1 indica ausencia de sesgo de escala)
      - Eliminación de heterocedasticidad remanente (Ljung-Box sobre z^2 y ARCH-LM de Engle)
    """
    r_t1 = panel["r_t1"].values
    rows = []

    for m in model_list:
        sigma_hat = np.sqrt(np.maximum(panel[f"s2_{m}"].values, 1e-12))
        z = r_t1 / sigma_hat
        z2 = z ** 2

        lb_z2 = acorr_ljungbox(z2, lags=[5, 10, 20], return_df=True)
        arch_5_stat, arch_5_p, _, _ = het_arch(z, nlags=5)
        arch_10_stat, arch_10_p, _, _ = het_arch(z, nlags=10)

        rows.append({
            "Modelo": m,
            "Media_z": float(np.mean(z)),
            "Varianza_z_Ideal_1": float(np.var(z, ddof=1)),
            "Asimetria_z": float(stats.skew(z)),
            "Exceso_Curtosis_z": float(stats.kurtosis(z, fisher=True)),
            "LB_z2_Lag5_p": float(lb_z2.loc[5, "lb_pvalue"]),
            "LB_z2_Lag10_p": float(lb_z2.loc[10, "lb_pvalue"]),
            "LB_z2_Lag20_p": float(lb_z2.loc[20, "lb_pvalue"]),
            "ARCH_LM_Lag5_Stat": float(arch_5_stat),
            "ARCH_LM_Lag5_p": float(arch_5_p),
            "ARCH_LM_Lag10_Stat": float(arch_10_stat),
            "ARCH_LM_Lag10_p": float(arch_10_p),
            "Sin_ARCH_Remanente_5pct": bool(arch_5_p > 0.05 and lb_z2.loc[10, "lb_pvalue"] > 0.05),
        })

    df_resid = pd.DataFrame(rows)
    df_resid.to_csv(RESULTS_DIR / "standardized_residuals_diagnostics.csv", index=False)
    return df_resid


# ==============================================================================
# BLOQUE 4: VISUALIZACIONES DE ESTRÉS EXTREMO Y CALIBRACIÓN DE RESIDUOS
# ==============================================================================
def plot_stress_regimes_and_residuals(
    panel: pd.DataFrame,
    df_regimes: pd.DataFrame,
    df_shocks: pd.DataFrame,
    egarch_x_name: str,
):
    # --------------------------------------------------------------------------
    # FIGURA 5: Superioridad en Estrés Extremo y Contención en Top-8 Crashes
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1.25]})

    # Panel A: Error medio QLIKE en regímenes de turbulencia y estrés extremo
    target_regs = [
        "1_Calma (< P50)",
        "2_Normal (P50 - P90)",
        "3_Turbulencia_Alta (P90 - P98)",
        "4_Estres_Extremo (> P98)",
        "8_Test_Ciego_2026 (Fase B)",
    ]
    short_labels = ["Calma\n(<P50)", "Normal\n(P50–P90)", "Turbulencia\n(P90–P98)", "Estrés Extremo\n(>P98)", "Test Ciego\n2026"]

    models_to_plot = [
        ("EGARCH(1,3)_GED", "#d62728"),
        (egarch_x_name, "#e7298a"),
        ("CNN_1D_Hybrid_2x2", "#ff7f0e"),
        ("GRU_Hybrid_2x2", "#2ca02c"),
        ("LSTM_Hybrid_2x2", "#1f77b4"),
    ]

    x = np.arange(len(target_regs))
    w = 0.16
    for idx, (m_name, col) in enumerate(models_to_plot):
        vals = []
        for reg in target_regs:
            sub = df_regimes[(df_regimes["Regimen"] == reg) & (df_regimes["Modelo"] == m_name)]
            vals.append(float(sub["Mean_QLIKE"].iloc[0]) if not sub.empty else np.nan)
        offset = (idx - 2) * w
        ax1.bar(x + offset, vals, width=w, label=m_name, color=col, alpha=0.88)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9.5)
    ax1.set_ylabel("Mean QLIKE (menor = mejor)")
    ax1.set_title("A. Desempeño por Régimen de Mercado\n(Ventaja Estructural de DL en Estrés Extremo > P98)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Panel B: Pérdida QLIKE puntual en los Top-8 peores shocks históricos
    top8 = df_shocks.head(8).copy()
    x8 = np.arange(len(top8))
    w8 = 0.22
    ax2.bar(x8 - 1.5 * w8, top8["QLIKE_EGARCH(1,3)_GED"], width=w8, color="#d62728", label="EGARCH(1,3)_GED")
    ax2.bar(x8 - 0.5 * w8, top8[f"QLIKE_{egarch_x_name}"], width=w8, color="#e7298a", label=egarch_x_name)
    ax2.bar(x8 + 0.5 * w8, top8["QLIKE_GRU_Hybrid_2x2"], width=w8, color="#2ca02c", label="GRU_Hybrid_2x2")
    ax2.bar(x8 + 1.5 * w8, top8["QLIKE_LSTM_Hybrid_2x2"], width=w8, color="#1f77b4", label="LSTM_Hybrid_2x2")

    shock_labels = [f"{r['Event_Date_t1']}\n({r['BTC_Ret_t1_pct']:+.0f}%)" for _, r in top8.iterrows()]
    ax2.set_xticks(x8)
    ax2.set_xticklabels(shock_labels, rotation=35, ha="right", fontsize=8.5)
    ax2.set_ylabel("Pérdida Diaria QLIKE en el Shock")
    ax2.set_title("B. Contención de Explosiones de Pérdida en los Top-8 Shocks\n(DL Amortigua el Impacto de Crashes Extremos)")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_regime_stress_superiority.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURA 6: Correlograma ACF de Residuos Estandarizados al Cuadrado z_{t+1}^2
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    r_t1 = panel["r_t1"].values
    max_lag = 25
    lags = np.arange(1, max_lag + 1)
    conf_band = 1.96 / np.sqrt(len(panel))

    acf_models = [
        ("Benchmark_Promedio", "#7f7f7f", ":"),
        ("EGARCH(1,3)_GED", "#d62728", "--"),
        (egarch_x_name, "#e7298a", "-."),
        ("GRU_Hybrid_2x2", "#2ca02c", "-"),
        ("LSTM_Hybrid_2x2", "#1f77b4", "-"),
    ]

    for m_name, col, ls in acf_models:
        z = r_t1 / np.sqrt(np.maximum(panel[f"s2_{m_name}"].values, 1e-12))
        acf_vals = acf(z ** 2, nlags=max_lag, fft=True)[1:]
        ax.plot(lags, acf_vals, marker="o", markersize=4, lw=1.6, ls=ls, color=col, label=f"{m_name} ($z^2$)")

    ax.axhline(conf_band, color="black", ls="--", lw=1.0, label="Banda de ruido blanco (95%)")
    ax.axhline(-conf_band, color="black", ls="--", lw=1.0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Rezago (días)")
    ax.set_ylabel(r"Autocorrelación ACF($z_{t+1}^2$)")
    ax.set_title("Diagnóstico de Calibración: Autocorrelación Remanente en Residuos Estandarizados al Cuadrado")
    ax.legend(loc="upper right", frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_standardized_residuals_acf.png")
    plt.close(fig)


def main():
    panel, model_list, egarch_x_name = build_unified_oos_panel()
    df_regimes = analyze_regimes_and_conditional_dm(panel, model_list, egarch_x_name)
    df_shocks = analyze_top_extreme_shocks(panel, model_list, top_n=15)
    df_resid = analyze_standardized_residuals(panel, model_list)
    plot_stress_regimes_and_residuals(panel, df_regimes, df_shocks, egarch_x_name)

    print("\n" + "=" * 120)
    print("1. ANÁLISIS POR REGÍMENES DE VOLATILIDAD Y TEST DIEBOLD-MARIANO CONDICIONAL:")
    print("=" * 120)
    disp_reg = [
        "Regimen", "Dias", "Modelo", "Mean_QLIKE", "Median_QLIKE", "Max_QLIKE_PeorDia",
        "WinRate_vs_GED_pct", "DM_vs_GED", "p_uni_vs_GED", "Sig_vs_GED", "p_uni_vs_EGARCHX", "Sig_vs_EGARCHX",
    ]
    print(df_regimes[disp_reg].to_string(index=False))

    print("\n" + "=" * 120)
    print("2. TOP-10 MAYORES SHOCKS DE VOLATILIDAD (2019–2026) — PÉRDIDA QLIKE POR MODELO:")
    print("=" * 120)
    shock_cols = [
        "Rank", "Event_Date_t1", "Fase", "BTC_Ret_t1_pct", "Realized_r2_pct2",
        "QLIKE_EGARCH(1,3)_GED", f"QLIKE_{egarch_x_name}", "QLIKE_CNN_1D_Hybrid_2x2",
        "QLIKE_GRU_Hybrid_2x2", "QLIKE_LSTM_Hybrid_2x2",
    ]
    print(df_shocks[shock_cols].head(10).to_string(index=False))

    print("\n" + "=" * 120)
    print("3. DIAGNÓSTICO DE RESIDUOS ESTANDARIZADOS z_{t+1} = r_{t+1} / sigma_{t+1}:")
    print("=" * 120)
    print(df_resid.to_string(index=False))

    print("\nArtefactos exportados en 'results/' y 'results/figures/':")
    print(f"  - {RESULTS_DIR / 'regime_error_analysis.csv'}")
    print(f"  - {RESULTS_DIR / 'top15_extreme_shocks_models.csv'}")
    print(f"  - {RESULTS_DIR / 'standardized_residuals_diagnostics.csv'}")
    print(f"  - {FIG_DIR / 'fig5_regime_stress_superiority.png'}")
    print(f"  - {FIG_DIR / 'fig6_standardized_residuals_acf.png'}")


if __name__ == "__main__":
    main()