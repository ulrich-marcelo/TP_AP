import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    PHASE_A_END_DATE,
    load_processed_data_with_dates,
    load_returns_reference,
)
from src.evaluation import compute_qlike_vector

RAW_DATA_PATH = ROOT / "data" / "raw" / "raw_data.csv"
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "figures" / "report_individual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 13.0,
    "axes.titlesize": 13.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 13.0,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "figure.dpi": 300,
})


def _save_single_fig(fig: plt.Figure, filename: str):
    out_path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


def _detect_egarch_x_name(df_base: pd.DataFrame) -> str:
    spec_path = RESULTS_DIR / "egarch_x_best_spec.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("spec") in df_base["model"].unique():
            return cfg["spec"]
    others = [m for m in df_base["model"].unique() if m not in ("Benchmark_Promedio", "EGARCH(1,3)_GED")]
    return str(others[0]) if others else "EGARCH(2,3)-X_GED"


def _shade_blind_test_2026(ax, max_date: pd.Timestamp):
    cutoff = pd.Timestamp(PHASE_A_END_DATE)
    if max_date > cutoff:
        ax.axvline(cutoff, color="#4b0082", ls="--", lw=1.5, alpha=0.85)
        ax.axvspan(cutoff, max_date, color="#e6e6fa", alpha=0.38, label="Fase B: Test Ciego 2026")


def _load_merged_oos_dataframe():
    df_base = pd.read_parquet(RESULTS_DIR / "baselines_predictions.parquet")
    df_base["Date"] = pd.to_datetime(df_base["Date"])
    df_arch = pd.read_parquet(RESULTS_DIR / "architecture_ensembles_predictions.parquet")
    df_arch["Date"] = pd.to_datetime(df_arch["Date"])

    egarch_x_name = _detect_egarch_x_name(df_base)

    comp = (
        df_base[df_base["model"] == "Benchmark_Promedio"][["Date", "phase", "fold", "y_true", "sigma2_pred", "qlike"]]
        .rename(columns={"sigma2_pred": "s2_prom", "qlike": "q_prom"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    for m_name, alias in [("EGARCH(1,3)_GED", "ged"), (egarch_x_name, "x")]:
        sub = df_base[df_base["model"] == m_name][["Date", "sigma2_pred", "qlike"]].rename(
            columns={"sigma2_pred": f"s2_{alias}", "qlike": f"q_{alias}"}
        )
        comp = pd.merge(comp, sub, on="Date", how="inner")

    for arch, alias in [("CNN_1D", "cnn"), ("GRU", "gru"), ("LSTM", "lstm")]:
        sub = df_arch[(df_arch["architecture"] == arch) & (df_arch["ensemble_type"] == "Hybrid_2x2")][
            ["Date", "sigma2_pred", "sigma2_pred_std", "qlike"]
        ].rename(columns={"sigma2_pred": f"s2_{alias}", "sigma2_pred_std": f"std_{alias}", "qlike": f"q_{alias}"})
        comp = pd.merge(comp, sub, on="Date", how="inner")

    comp["s2_super"] = (comp["s2_cnn"] + comp["s2_gru"] + comp["s2_lstm"]) / 3.0
    comp["q_super"] = compute_qlike_vector(comp["y_true"].values, comp["s2_super"].values)
    return comp, egarch_x_name


# ==============================================================================
# INSTANCIA 1: GRÁFICOS INDIVIDUALES DEL EDA (FIGURAS 00 A 04)
# ==============================================================================
def generate_eda_individual_plots():
    print("\n[Instancia 1/3] Generando gráficos individuales de EDA (00 a 04)...")
    df_proc, _, _, _, _ = load_processed_data_with_dates(DATA_PATH, phase="phase_a")
    df_ref = load_returns_reference(DATA_PATH, phase="phase_a")

    df = df_proc.copy()
    df["BTC_r_cc"] = df_ref["BTC_r_cc"]

    # 00. Serie de Precios Crudos (BTC y SPY) — Justificación visual de No Estacionariedad
    if RAW_DATA_PATH.exists():
        df_raw = pd.read_csv(RAW_DATA_PATH)
        if "Date" in df_raw.columns:
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            df_raw = df_raw.set_index("Date").sort_index()
        else:
            df_raw.index = pd.to_datetime(df_raw.index)
            df_raw = df_raw.sort_index()

        df_raw_a = df_raw.loc[df.index.min() : df.index.max()].copy()

        if "BTC_Close" in df_raw_a.columns:
            fig, ax1 = plt.subplots(figsize=(9.0, 4.7))
            btc_close = df_raw_a["BTC_Close"].ffill()

            l1 = ax1.plot(
                df_raw_a.index, btc_close,
                color="#1f77b4", lw=1.8, label="Precio Cierre Bitcoin (BTC_Close, USD — eje izq.)"
            )
            ax1.fill_between(df_raw_a.index, 0, btc_close, color="#1f77b4", alpha=0.10)
            ax1.set_ylabel("Precio de Bitcoin (USD)", color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
            ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax1.set_ylim(0, btc_close.max() * 1.18)

            lines = l1
            if "SPY_Close" in df_raw_a.columns:
                ax2 = ax1.twinx()
                spy_close = df_raw_a["SPY_Close"].ffill()
                l2 = ax2.plot(
                    df_raw_a.index, spy_close,
                    color="#ff7f0e", lw=1.5, ls="--", alpha=0.85,
                    label="Precio Cierre S&P 500 ETF (SPY_Close, USD — eje der.)"
                )
                ax2.set_ylabel("Precio de SPY (USD)", color="#d95f02")
                ax2.tick_params(axis="y", labelcolor="#d95f02")
                ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
                ax2.set_ylim(0, spy_close.max() * 1.18)
                ax2.grid(False)
                lines = l1 + l2

            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper left", frameon=True, fontsize=9.8)
            ax1.set_title(
                "Series de Precios Crudos (2016–2025):\n"
                "Evidencia de No Estacionariedad y Cambio de Escala"
            )
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            _save_single_fig(fig, "00_eda_raw_prices_nonstationarity.png")

    # 01. Volatilidad móvil diaria a 30 días (%)
    vol_30d_daily = df["BTC_r_cc"].rolling(30).std() * 100.0
    med_vol = float(vol_30d_daily.median())

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(df.index, vol_30d_daily, color="#1f77b4", lw=1.6, label="Volatilidad móvil diaria 30d (%)")
    ax.axhline(med_vol, color="#333333", ls="--", lw=1.2, label=f"Mediana histórica ({med_vol:.2f}% diario)")
    ax.fill_between(
        df.index, med_vol, vol_30d_daily,
        where=(vol_30d_daily > med_vol),
        color="#d62728", alpha=0.25, interpolate=True,
        label="Clústeres de alta volatilidad (> Mediana)",
    )
    ax.fill_between(
        df.index, 0, vol_30d_daily,
        where=(vol_30d_daily <= med_vol),
        color="#2ca02c", alpha=0.14, interpolate=True,
        label="Regímenes de baja volatilidad (≤ Mediana)",
    )
    ax.set_ylim(0, vol_30d_daily.max() * 1.15)
    ax.set_ylabel("Volatilidad diaria móvil (%)")
    ax.set_title(
        "Volatility Clustering en Bitcoin (2016–2025):\n"
        "Alternancia de Regímenes de Baja y Alta Volatilidad"
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_single_fig(fig, "01_eda_btc_returns_volatility_envelope.png")

    # 02. Correlograma ACF comparativo r_t vs |r_t| vs r_t^2
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    max_lag = 30
    lags = np.arange(1, max_lag + 1)
    acf_r = acf(df["BTC_r_cc"].values, nlags=max_lag, fft=True)[1:]
    acf_abs = acf(np.abs(df["BTC_r_cc"].values), nlags=max_lag, fft=True)[1:]
    acf_r2 = acf(df["BTC_r_cc"].values ** 2, nlags=max_lag, fft=True)[1:]
    conf_band = 1.96 / np.sqrt(len(df))

    w = 0.28
    ax.bar(
        lags - w, acf_r, width=w,
        color="#1a1a1a", edgecolor="#000000", linewidth=0.35, zorder=3,
        label=r"Retorno con signo $r_t$ (ruido blanco)",
    )
    ax.bar(
        lags, acf_abs, width=w,
        color="#1f77b4", alpha=0.88, zorder=2,
        label=r"Módulo $|r_t|$ (señal EGARCH y entradas DL)",
    )
    ax.bar(
        lags + w, acf_r2, width=w,
        color="#d62728", alpha=0.88, zorder=2,
        label=r"Varianza realizada $r_t^2$ (efecto ARCH)",
    )
    ax.axhline(conf_band, color="black", ls="--", lw=1.1, label="Umbral ruido blanco (95%)")
    ax.axhline(-conf_band, color="black", ls="--", lw=1.1)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Rezago (días) — Justificación de ventanas de entrada de 5 a 21 días")
    ax.set_ylabel("Autocorrelación (ACF)")
    ax.set_title(
        "Memoria Temporal: Retornos Direccionales (Ruido Blanco)\n"
        "vs. Magnitud de Volatilidad",
        pad=44,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=True, ncol=2, fontsize=9.8)
    _save_single_fig(fig, "02_eda_acf_returns_vs_variance.png")

    # 03. News Impact Curve empírica
    df["BTC_Decil"] = pd.qcut(df["BTC_r_oc"], q=10, labels=[f"D{d}" for d in range(1, 11)])
    nic = df.groupby("BTC_Decil", observed=False).agg(
        BTC_r_oc_Medio_pct=("BTC_r_oc", lambda s: float(s.mean() * 100.0)),
        Target_r2_Media_pct2=("target_r2_t1", lambda s: float(s.mean() * 10000.0)),
        Vol_Diaria_pct=("target_r2_t1", lambda s: float(np.sqrt(s.mean()) * 100.0)),
    ).reset_index()

    mean_d2_d10 = float(nic.loc[1:, "Target_r2_Media_pct2"].mean())

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x_pos = np.arange(len(nic))
    bar_colors = ["#d62728" if i == 0 else "#4c78a8" for i in range(len(nic))]
    bars = ax.bar(x_pos, nic["Target_r2_Media_pct2"], color=bar_colors, alpha=0.88, edgecolor="black", linewidth=0.6)

    for bar, (_, row) in zip(bars, nic.iterrows()):
        h_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val + 0.55,
            f"{h_val:.1f} %²\n({row['Vol_Diaria_pct']:.2f}%/d)",
            ha="center", va="bottom", fontsize=9.2, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.0),
        )

    d1_val = float(nic.loc[0, "Target_r2_Media_pct2"])
    ax.axhline(mean_d2_d10, color="#333333", ls="--", lw=1.3, label=f"Promedio deciles D2–D10 ({mean_d2_d10:.1f} %²)")
    legend_patches = [
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.88,
              label=f"D1 — Desplomes severos: duplica la varianza futura ({d1_val:.1f} %²)"),
        Patch(facecolor="#4c78a8", edgecolor="black", alpha=0.88, label="D2 a D10 — Resto de escenarios de mercado"),
        plt.Line2D([0], [0], color="#333333", ls="--", lw=1.3, label=f"Promedio D2–D10 ({mean_d2_d10:.1f} %²)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True, fontsize=9.8)
    ax.set_ylim(0, nic["Target_r2_Media_pct2"].max() * 1.35)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}\n({m:+.1f}%)" for d, m in zip(nic["BTC_Decil"], nic["BTC_r_oc_Medio_pct"])], fontsize=10.5)
    ax.set_xlabel("Decil de retorno diario BTC (retorno medio del decil)")
    ax.set_ylabel(r"Varianza media al día siguiente $\mathbb{E}[r_{t+1}^2]$ (%$^2$)")
    ax.set_title(
        "News Impact Curve Empírica:\n"
        "Efecto Apalancamiento Asimétrico ante Caídas Severas (D1)"
    )
    _save_single_fig(fig, "03_eda_news_impact_curve_deciles.png")

    # 04. QQ-Plot anotado de colas pesadas
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    r_vals = df["BTC_r_cc"].values
    z_emp = (r_vals - np.mean(r_vals)) / np.std(r_vals, ddof=1)
    (osm, osr), _ = stats.probplot(z_emp, dist="norm")

    ax.axvspan(-1.5, 1.5, color="#2ca02c", alpha=0.10, label="Zona central (~87% días: cercano a Normal)")
    ax.scatter(osm, osr, s=16, alpha=0.65, color="#1f77b4", label="Retornos diarios reales de Bitcoin")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, line_x, color="#d62728", lw=2.0, ls="--", label="Distribución Normal teórica N(0, 1)")

    ax.annotate(
        f"Cola izquierda extrema ({osr.min():.1f}σ)\nCrashes mucho más severos\nque una curva Normal",
        xy=(osm.min(), osr.min()),
        xytext=(-2.2, -11.5),
        fontsize=10.5,
        fontweight="bold",
        color="#8c1515",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f5", edgecolor="#d62728"),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax.annotate(
        f"Cola derecha extrema (+{osr.max():.1f}σ)\nRallies explosivos",
        xy=(osm.max(), osr.max()),
        xytext=(-0.2, 4.4),
        fontsize=10.5,
        fontweight="bold",
        color="#8c1515",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f5", edgecolor="#d62728"),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    ax.set_ylim(-14.5, 8.8)
    ax.set_xlabel("Desvíos estándar esperados bajo Normalidad N(0, 1)")
    ax.set_ylabel("Desvíos estándar reales observados en Bitcoin")
    ax.set_title(
        "QQ-Plot de Retornos de Bitcoin:\n"
        "Prueba Visual de Colas Pesadas (Leptocurtosis)"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9.8)
    _save_single_fig(fig, "04_eda_qqplot_fat_tails.png")


# ==============================================================================
# INSTANCIA 2: COMPARACIÓN GENERAL (FIGURA 05)
# ==============================================================================
def generate_comparison_individual_plots():
    print("\n[Instancia 2/3] Generando gráfico individual comparativo (05)...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    max_date = comp["Date"].max()

    # 05. Diferencial Acumulado de Pérdida QLIKE
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.plot(
        comp["Date"], (comp["q_prom"] - comp["q_super"]).cumsum(),
        label="Benchmark Promedio vs. DL SuperEnsemble (60M)", color="#7f7f7f", lw=1.9, ls=":"
    )
    ax.plot(
        comp["Date"], (comp["q_ged"] - comp["q_super"]).cumsum(),
        label="EGARCH(1,3)_GED vs. DL SuperEnsemble (60M)", color="#d62728", lw=2.2
    )
    ax.plot(
        comp["Date"], (comp["q_x"] - comp["q_super"]).cumsum(),
        label=f"{egarch_x_name} vs. DL SuperEnsemble (60M)", color="#1f77b4", lw=2.3
    )
    _shade_blind_test_2026(ax, max_date)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"Diferencial Acumulado $\sum (QLIKE_{\mathrm{base}} - QLIKE_{\mathrm{DL}})$")
    ax.set_title(
        "Evolución Acumulada del Diferencial de Pérdida Out-of-Sample (2019–2026)\n"
        "(Ventaja en Crisis 2019–2024; Retroceso en 2025-S2 y Empate en 2026)",
        pad=44,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=True, ncol=2, fontsize=9.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _save_single_fig(fig, "05_comp_cumulative_loss_differential.png")


# ==============================================================================
# INSTANCIA 3: REGÍMENES DE ESTRÉS EXTREMO Y CRISIS COVID-19 (FIGURAS 06 A 08)
# ==============================================================================
def generate_diagnostics_individual_plots():
    print("\n[Instancia 3/3] Generando gráficos individuales de regímenes y estrés extremo (06 a 08)...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    reg_path = RESULTS_DIR / "regime_error_analysis.csv"
    shocks_path = RESULTS_DIR / "top15_extreme_shocks_models.csv"

    if not reg_path.exists() or not shocks_path.exists():
        raise FileNotFoundError("Ejecutá primero scripts/10_error_analysis_and_diagnostics.py.")

    df_reg = pd.read_csv(reg_path)
    df_shk = pd.read_csv(shocks_path)

    # 06. Reducción Porcentual Relativa (%) respecto a EGARCH(1,3)_GED por Régimen de Volatilidad
    target_regs = [
        "1_Baja_Volatilidad (< P50)",
        "2_Volatilidad_Moderada (P50 - P90)",
        "3_Alta_Volatilidad (P90 - P98)",
        "4_Estres_Extremo (> P98)",
        "8_Test_Ciego_2026 (Fase B)",
    ]
    short_labels = [
        "Baja Volatilidad\n(<P50)",
        "Vol. Moderada\n(P50–P90)",
        "Alta Volatilidad\n(P90–P98)",
        "Estrés Extremo\n(>P98)",
        "Test Ciego\n2026",
    ]
    models_rel = [
        (egarch_x_name, "#e7298a"),
        ("CNN_1D_Hybrid_2x2", "#ff7f0e"),
        ("GRU_Hybrid_2x2", "#2ca02c"),
        ("LSTM_Hybrid_2x2", "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    x = np.arange(len(target_regs))
    w = 0.19

    for idx, (m_name, col) in enumerate(models_rel):
        imp_vals = []
        sigs = []
        for reg in target_regs:
            ged_val = float(df_reg[(df_reg["Regimen"] == reg) & (df_reg["Modelo"] == "EGARCH(1,3)_GED")]["Mean_QLIKE"].iloc[0])
            sub_m = df_reg[(df_reg["Regimen"] == reg) & (df_reg["Modelo"] == m_name)].iloc[0]
            m_val = float(sub_m["Mean_QLIKE"])
            imp_vals.append((ged_val - m_val) / ged_val * 100.0)
            sigs.append(str(sub_m["Sig_vs_GED"]))

        offset = (idx - 1.5) * w
        bars = ax.bar(x + offset, imp_vals, width=w, label=m_name, color=col, alpha=0.88)
        for bar, sig in zip(bars, sigs):
            hv = bar.get_height()
            if sig in ("***", "**", "*"):
                ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.6, "*", ha="center", va="bottom", fontsize=13.5, fontweight="bold")
            elif sig == "·":
                ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.6, "†", ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    ax.axhline(0, color="black", lw=1.1, label="Baseline: EGARCH(1,3)_GED (0%)")
    ax.plot([], [], " ", label="* p < 0.05  |  † p < 0.10 (DM unilateral)")
    ax.set_ylim(-9, 33)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10.8)
    ax.set_ylabel("Reducción porcentual de Mean QLIKE vs. EGARCH(1,3)_GED (%)")
    ax.set_title(
        "Ventaja Condicional por Régimen de Volatilidad\n"
        "(Positivo = Menor Error que EGARCH(1,3)_GED)",
        pad=54,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=True, ncol=3, fontsize=9.4)
    _save_single_fig(fig, "06_diag_qlike_by_volatility_regime.png")

    # 07. Promedio de QLIKE y Peor Error Máximo en los Top-15 Shocks Históricos
    models_shk = [
        ("Benchmark Prom.", "QLIKE_Benchmark_Promedio", "#7f7f7f"),
        ("EGARCH(1,3)_GED", "QLIKE_EGARCH(1,3)_GED", "#d62728"),
        (egarch_x_name, f"QLIKE_{egarch_x_name}", "#e7298a"),
        ("CNN 1D (Hybrid)", "QLIKE_CNN_1D_Hybrid_2x2", "#ff7f0e"),
        ("GRU (Hybrid)", "QLIKE_GRU_Hybrid_2x2", "#2ca02c"),
        ("LSTM (Hybrid)", "QLIKE_LSTM_Hybrid_2x2", "#1f77b4"),
        ("DL SuperEns. (60M)", "QLIKE_DL_SuperEnsemble_60M", "#9467bd"),
    ]
    mean_top15 = [float(df_shk[m[1]].mean()) for m in models_shk]
    max_top15 = [float(df_shk[m[1]].max()) for m in models_shk]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.2, 4.8))
    xs = np.arange(len(models_shk))
    cols_s = [m[2] for m in models_shk]
    lbls_s = [m[0] for m in models_shk]

    b1 = ax_l.bar(xs, mean_top15, color=cols_s, alpha=0.88, edgecolor="black", linewidth=0.6)
    for bar in b1:
        hv = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.5, f"{hv:.2f}", ha="center", va="bottom", fontsize=9.8, fontweight="bold")
    ax_l.set_ylim(0, max(mean_top15) * 1.18)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(lbls_s, rotation=32, ha="right", fontsize=10.0)
    ax_l.set_ylabel("Promedio de QLIKE en los 15 shocks")
    ax_l.set_title("A. Error Promedio en Top-15 Shocks\n(DL SuperEns. −32.7% vs. EGARCH_GED)", fontsize=12.5)

    b2 = ax_r.bar(xs, max_top15, color=cols_s, alpha=0.88, edgecolor="black", linewidth=0.6)
    for bar in b2:
        hv = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width() / 2.0, hv + 3.0, f"{hv:.1f}", ha="center", va="bottom", fontsize=9.8, fontweight="bold")
    ax_r.set_ylim(0, max(max_top15) * 1.18)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(lbls_s, rotation=32, ha="right", fontsize=10.0)
    ax_r.set_ylabel("Peor Pérdida Diaria (Max QLIKE)")
    ax_r.set_title("B. Peor Explosión Individual\n(187.7 en EGARCH_GED vs. 125.8 en DL)", fontsize=12.5)

    _save_single_fig(fig, "07_diag_top15_shocks_aggregate.png")

    # 08. Reacción Post-Shock en Crisis COVID-19 en escala %^2 (Varianza realizada suavizada 7d)
    c_covid = comp[(comp["Date"] >= "2020-02-01") & (comp["Date"] <= "2020-05-31")].copy()
    var_real_7d = (c_covid["y_true"] * 10000.0).rolling(7, min_periods=1).mean()
    s2_ged_pct2 = c_covid["s2_ged"] * 10000.0
    s2_x_pct2 = c_covid["s2_x"] * 10000.0
    s2_super_pct2 = c_covid["s2_super"] * 10000.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9.4, 5.2),
        gridspec_kw={"height_ratios": [1.0, 3.0], "hspace": 0.08}
    )

    for ax in (ax_top, ax_bot):
        ax.fill_between(
            c_covid["Date"], 0, var_real_7d,
            color="#bdbdbd", alpha=0.38, label="Varianza realizada móvil 7d (%²)"
        )
        ax.plot(c_covid["Date"], s2_ged_pct2, color="#d62728", lw=1.8, ls="--", label="EGARCH(1,3)_GED")
        ax.plot(c_covid["Date"], s2_x_pct2, color="#e7298a", lw=1.8, ls="-.", label=egarch_x_name)
        ax.plot(c_covid["Date"], s2_super_pct2, color="#1f77b4", lw=2.3, label="DL SuperEnsemble (60M)")

    ax_top.set_ylim(215, 1720)
    ax_bot.set_ylim(-5, 195)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.011
    kw = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d * 3.0, +d * 3.0), **kw)
    ax_top.plot((1 - d, 1 + d), (-d * 3.0, +d * 3.0), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_top.set_title(
        "Reacción Post-Shock en la Crisis COVID-19 (Febrero – Mayo 2020)\n"
        "en Escala %² (Eje Y Recortado)",
        pad=44,
    )
    fig.supylabel(r"Varianza pronosticada $\hat{\sigma}^2_{t+1}$ (%$^2$)", x=0.01, fontsize=13.0)
    ax_top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=True, ncol=2, fontsize=10.0)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    out_path_08 = OUT_DIR / "08_pred_covid_regime_response.png"
    fig.savefig(out_path_08, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path_08.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de los 9 Gráficos Individuales del Informe (300 DPI)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "eda", "comparison", "diagnostics"],
        help="Instancia de gráficos a generar (por defecto 'all').",
    )
    args = parser.parse_args()

    print(f"Exportando figuras individuales (1 gráfico = 1 PNG) en: {OUT_DIR.resolve()}")
    if args.stage in ("all", "eda"):
        generate_eda_individual_plots()
    if args.stage in ("all", "comparison"):
        generate_comparison_individual_plots()
    if args.stage in ("all", "diagnostics"):
        generate_diagnostics_individual_plots()

    print("\nGeneración finalizada.")