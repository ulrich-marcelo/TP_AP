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
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
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

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "figures" / "report_individual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11.5,
    "axes.titlesize": 13.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
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
# INSTANCIA 1: GRÁFICOS INDIVIDUALES DEL EDA (FIGURAS 01 A 06)
# ==============================================================================
def generate_eda_individual_plots():
    print("\n[Instancia 1/3] Generando gráficos individuales de EDA (01 a 06)...")
    df_proc, X, y, _, feature_names = load_processed_data_with_dates(DATA_PATH, phase="phase_a")
    df_ref = load_returns_reference(DATA_PATH, phase="phase_a")

    df = df_proc.copy()
    df["BTC_r_cc"] = df_ref["BTC_r_cc"]

    # 01. Volatilidad móvil diaria a 30 días (%)
    vol_30d_daily = df["BTC_r_cc"].rolling(30).std() * 100.0
    med_vol = float(vol_30d_daily.median())

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
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
        label="Regímenes de calma (≤ Mediana)",
    )
    ax.set_ylim(0, vol_30d_daily.max() * 1.12)
    ax.set_ylabel("Volatilidad diaria móvil (%)")
    ax.set_title("Volatility Clustering en Bitcoin: Alternancia de Regímenes de Calma y Estrés (2016–2025)")
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_single_fig(fig, "01_eda_btc_returns_volatility_envelope.png")

    # 02. Correlograma ACF comparativo r_t vs |r_t| vs r_t^2
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    max_lag = 30
    lags = np.arange(1, max_lag + 1)
    acf_r = acf(df["BTC_r_cc"].values, nlags=max_lag, fft=True)[1:]
    acf_abs = acf(np.abs(df["BTC_r_cc"].values), nlags=max_lag, fft=True)[1:]
    acf_r2 = acf(df["BTC_r_cc"].values ** 2, nlags=max_lag, fft=True)[1:]
    conf_band = 1.96 / np.sqrt(len(df))

    w = 0.27
    ax.bar(lags - w, acf_r, width=w, color="#7f7f7f", label=r"Retorno con signo $r_t$ (sin memoria direccional)")
    ax.bar(lags, acf_abs, width=w, color="#1f77b4", label=r"Módulo $|r_t|$ (señal usada en EGARCH y entradas DL)")
    ax.bar(lags + w, acf_r2, width=w, color="#d62728", label=r"Varianza realizada $r_t^2$ (efecto ARCH)")
    ax.axhline(conf_band, color="black", ls="--", lw=1.1, label="Umbral ruido blanco (95%)")
    ax.axhline(-conf_band, color="black", ls="--", lw=1.1)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Rezago (días) — Justificación de ventanas de entrada de 7 a 21 días")
    ax.set_ylabel("Autocorrelación (ACF)")
    ax.set_title("Memoria Temporal: Retornos Direccionales (Ruido Blanco) vs. Magnitud de Volatilidad")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "02_eda_acf_returns_vs_variance.png")

    # 03. News Impact Curve empírica
    df["BTC_Decil"] = pd.qcut(df["BTC_r_oc"], q=10, labels=[f"D{d}" for d in range(1, 11)])
    nic = df.groupby("BTC_Decil", observed=False).agg(
        BTC_r_oc_Medio_pct=("BTC_r_oc", lambda s: float(s.mean() * 100.0)),
        Target_r2_Media_pct2=("target_r2_t1", lambda s: float(s.mean() * 10000.0)),
        Vol_Diaria_pct=("target_r2_t1", lambda s: float(np.sqrt(s.mean()) * 100.0)),
    ).reset_index()

    mean_d2_d10 = float(nic.loc[1:, "Target_r2_Media_pct2"].mean())

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x_pos = np.arange(len(nic))
    bar_colors = ["#d62728" if i == 0 else "#4c78a8" for i in range(len(nic))]
    bars = ax.bar(x_pos, nic["Target_r2_Media_pct2"], color=bar_colors, alpha=0.88, edgecolor="black", linewidth=0.6)

    for bar, (_, row) in zip(bars, nic.iterrows()):
        h_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val + 0.45,
            f"{h_val:.1f} %²\n({row['Vol_Diaria_pct']:.2f}%/d)",
            ha="center", va="bottom", fontsize=8.6, fontweight="bold",
        )

    ax.axhline(mean_d2_d10, color="#333333", ls="--", lw=1.3, label=f"Promedio deciles D2–D10 ({mean_d2_d10:.1f} %²)")
    legend_patches = [
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.88, label="D1 — Desplomes severos: duplica la varianza futura (23.4 %²)"),
        Patch(facecolor="#4c78a8", edgecolor="black", alpha=0.88, label="D2 a D10 — Resto de escenarios de mercado"),
        plt.Line2D([0], [0], color="#333333", ls="--", lw=1.3, label=f"Promedio D2–D10 ({mean_d2_d10:.1f} %²)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True)
    ax.set_ylim(0, nic["Target_r2_Media_pct2"].max() * 1.25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}\n({m:+.1f}%)" for d, m in zip(nic["BTC_Decil"], nic["BTC_r_oc_Medio_pct"])])
    ax.set_xlabel("Decil de retorno diario BTC (retorno medio del decil)")
    ax.set_ylabel(r"Varianza media al día siguiente $\mathbb{E}[r_{t+1}^2]$ (%$^2$)")
    ax.set_title("News Impact Curve Empírica: Efecto Apalancamiento Asimétrico ante Caídas Severas (D1)")
    _save_single_fig(fig, "03_eda_news_impact_curve_deciles.png")

    # 04. Señal Lineal |Pearson| vs Información Mutua
    mi_raw = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
    rows_nl = []
    for i, col in enumerate(feature_names):
        r_p, _ = stats.pearsonr(X[:, i], y)
        rows_nl.append({"Feature": col, "Abs_Pearson": abs(float(r_p)), "MI": float(mi_raw[i])})
    df_nl = pd.DataFrame(rows_nl).sort_values("MI", ascending=True)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    y_pos = np.arange(len(df_nl))
    p_norm = df_nl["Abs_Pearson"] / max(df_nl["Abs_Pearson"].max(), 1e-8)
    mi_norm = df_nl["MI"] / max(df_nl["MI"].max(), 1e-8)
    h = 0.36
    ax.barh(y_pos - h / 2, p_norm, height=h, color="#7f7f7f", label=r"$|r_{\mathrm{Pearson}}|$ (dependencia lineal, norm.)")
    ax.barh(y_pos + h / 2, mi_norm, height=h, color="#2ca02c", label="Información Mutua (dependencia no lineal, norm.)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_nl["Feature"])
    ax.set_xlabel("Intensidad relativa de señal predictiva (normalizada [0, 1])")
    ax.set_title("Justificación de Deep Learning: Señal Lineal vs. Información Mutua No Lineal")
    ax.legend(loc="lower right", frameon=True)
    _save_single_fig(fig, "04_eda_linear_vs_nonlinear_signal.png")

    # 05. Línea temporal de Outliers Extremos con eje Y recortado (A-E sin roces)
    y_pct2 = df["target_r2_t1"] * 10000.0
    q99_pct2 = float(y_pct2.quantile(0.99))
    next_dates = pd.Series(df.index, index=df.index).shift(-1).fillna(
        pd.Series(df.index + pd.Timedelta(days=1), index=df.index)
    )
    r_t1 = df["BTC_r_cc"].shift(-1).fillna(
        np.sign(df["BTC_r_oc"].shift(-1)).fillna(1.0) * np.sqrt(df["target_r2_t1"])
    )

    df_all_events = pd.DataFrame({
        "Event_Date": pd.to_datetime(next_dates.values),
        "y_pct2": y_pct2.values,
        "ret_pct": r_t1.values * 100.0,
    }).sort_values("Event_Date").reset_index(drop=True)

    top_clean = []
    for _, row in df_all_events.sort_values("y_pct2", ascending=False).iterrows():
        if all(abs((row["Event_Date"] - prev["Event_Date"]).days) >= 30 for prev in top_clean):
            top_clean.append(row)
        if len(top_clean) == 5:
            break
    top_clean = sorted(top_clean, key=lambda r: r["Event_Date"])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 5.8),
        gridspec_kw={"height_ratios": [1.0, 3.3], "hspace": 0.08}
    )

    for ax in (ax_top, ax_bot):
        ax.plot(
            df_all_events["Event_Date"], df_all_events["y_pct2"],
            color="#1f77b4", lw=0.95, alpha=0.85,
            label=r"Varianza realizada diaria $r_{t+1}^2$ (%$^2$)"
        )

    ax_bot.axhline(q99_pct2, color="#ff7f0e", ls="--", lw=1.4, label=f"Percentil 99 ({q99_pct2:.1f} %²)")
    ax_top.set_ylim(1850, 2320)
    ax_bot.set_ylim(-15, 585)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d * 3.3, +d * 3.3), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 3.3, +d * 3.3), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # B desplazado a la izquierda (-16, 22) para no rozar la línea vertical de C
    letter_offsets = [(-22, 14), (-16, 22), (0, 20), (18, 16), (20, 0)]
    for idx, row in enumerate(top_clean):
        letter = chr(65 + idx)
        ev_dt = row["Event_Date"]
        y_val = row["y_pct2"]
        dx, dy = letter_offsets[idx]
        target_ax = ax_top if y_val > 1000 else ax_bot
        target_ax.scatter([ev_dt], [y_val], color="#d62728", s=55, zorder=5)
        target_ax.annotate(
            letter,
            xy=(ev_dt, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor="#d62728", edgecolor="black", lw=0.8),
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.0),
        )

    ax_top.set_title("Identificación de los 5 Principales Shocks de Volatilidad en Bitcoin (A–E, Eje Y Recortado)", pad=10)
    fig.supylabel(r"Varianza realizada $r_{t+1}^2$ (%$^2$)", x=0.015, fontsize=12)
    ax_bot.legend(loc="upper right", frameon=True)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    out_path_05 = OUT_DIR / "05_eda_outliers_timeline.png"
    fig.savefig(out_path_05, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path_05.name}")

    # 06. QQ-Plot anotado de colas pesadas
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    r_vals = df["BTC_r_cc"].values
    z_emp = (r_vals - np.mean(r_vals)) / np.std(r_vals, ddof=1)
    (osm, osr), _ = stats.probplot(z_emp, dist="norm")

    ax.axvspan(-1.5, 1.5, color="#2ca02c", alpha=0.10, label="Zona central (~87% días: comportamiento cercano a Normal)")
    ax.scatter(osm, osr, s=15, alpha=0.65, color="#1f77b4", label="Retornos diarios reales de Bitcoin")
    line_x = np.array([osm.min(), osm.max()])
    ax.plot(line_x, line_x, color="#d62728", lw=2.0, ls="--", label="Distribución Normal teórica N(0, 1)")

    ax.annotate(
        f"Cola izquierda extrema ({osr.min():.1f}σ)\nCrashes mucho más severos\nque una curva Normal",
        xy=(osm.min(), osr.min()),
        xytext=(-2.1, -11.2),
        fontsize=9.2,
        fontweight="bold",
        color="#8c1515",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f5", edgecolor="#d62728"),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax.annotate(
        f"Cola derecha extrema (+{osr.max():.1f}σ)\nRallies explosivos",
        xy=(osm.max(), osr.max()),
        xytext=(0.15, 4.1),
        fontsize=9.2,
        fontweight="bold",
        color="#8c1515",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f5", edgecolor="#d62728"),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    ax.set_ylim(-14.5, 8.8)
    ax.set_xlabel("Desvíos estándar esperados bajo Normalidad N(0, 1)")
    ax.set_ylabel("Desvíos estándar reales observados en Bitcoin")
    ax.set_title("QQ-Plot de Retornos de Bitcoin: Prueba Visual de Colas Pesadas (Leptocurtosis)")
    ax.legend(loc="upper left", frameon=True, fontsize=9.2)
    _save_single_fig(fig, "06_eda_qqplot_fat_tails.png")


# ==============================================================================
# INSTANCIA 2: COMPARACIÓN GENERAL Y TEST CIEGO 2026 (FIGURAS 07 A 09)
# ==============================================================================
def generate_comparison_individual_plots():
    print("\n[Instancia 2/3] Generando gráficos individuales comparativos (07 a 09)...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    max_date = comp["Date"].max()

    # 07. Diferencial Acumulado de Pérdida QLIKE
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
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
    ax.set_title("Evolución Acumulada del Diferencial de Pérdida Out-of-Sample (2019–2026)\n(Ventaja Acumulada en Crisis 2019–2024; Retroceso en 2025-S2 y Empate en 2026)")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _save_single_fig(fig, "07_comp_cumulative_loss_differential.png")

    # 08. Evolución Temporal por Semestres y Etapas Estructurales
    fold_sum = (
        comp.groupby(["fold", "phase"], as_index=False)
        .agg(
            start_date=("Date", "min"),
            end_date=("Date", "max"),
            q_prom=("q_prom", "mean"),
            q_ged=("q_ged", "mean"),
            q_x=("q_x", "mean"),
            q_super=("q_super", "mean"),
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )

    def _format_fold_label(row):
        mid_dt = row["start_date"] + (row["end_date"] - row["start_date"]) / 2
        sem = "S1" if mid_dt.month <= 6 else "S2"
        suffix = "*" if row["phase"] == "Fase_B_Test_Ciego" else ""
        return f"{mid_dt.year}-{sem}{suffix}"

    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    x_idx = np.arange(len(fold_sum))

    ax.plot(x_idx, fold_sum["q_prom"], marker="s", markersize=5.5, lw=1.6, ls=":", color="#7f7f7f", label="Benchmark Promedio")
    ax.plot(x_idx, fold_sum["q_ged"], marker="^", markersize=6.5, lw=2.0, ls="--", color="#d62728", label="EGARCH(1,3)_GED")
    ax.plot(x_idx, fold_sum["q_x"], marker="d", markersize=6.0, lw=2.0, ls="-.", color="#e7298a", label=egarch_x_name)
    ax.plot(x_idx, fold_sum["q_super"], marker="o", markersize=6.8, lw=2.4, ls="-", color="#1f77b4", label="DL SuperEnsemble (60M)")

    eras = [
        (-0.5, 3.5, "#fde0dd", "Etapa 1 (2019–2020)\nEstrés y Shock COVID"),
        (3.5, 9.5, "#f0f0f0", "Etapa 2 (2021–2023)\nCiclo Retail y Colapso FTX"),
        (9.5, 13.5, "#e5f5e0", "Etapa 3A (2024–2025)\nMercado Institucional ETF"),
        (13.5, len(fold_sum) - 0.5, "#e6e6fa", "Etapa 3B (2026*)\nTest Ciego"),
    ]
    for x0, x1, col_bg, era_txt in eras:
        ax.axvspan(x0, x1, color=col_bg, alpha=0.45, zorder=0)
        if x0 > 0:
            ax.axvline(x0, color="#666666", ls=":", lw=1.1)
        ax.text((x0 + x1) / 2.0, 3.48, era_txt, ha="center", va="top", fontsize=8.8, fontweight="bold", color="#333333")

    ax.set_ylim(1.25, 3.62)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_format_fold_label(r) for _, r in fold_sum.iterrows()], rotation=35, ha="right", fontsize=9.2)
    ax.set_ylabel("QLIKE Medio del Semestre")
    ax.set_title("Evolución Semestre a Semestre por Etapa del Mercado (2019–2026)\n(Maduración Estructural del Error y Alternancia según Régimen de Volatilidad)")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "08_comp_era_evolution_lines.png")

    # 09. Test Ciego 2026: Comparación contra el Mejor Baseline Econométrico (EGARCH(2,3)-X_GED)
    sub_b = comp[comp["phase"] == "Fase_B_Test_Ciego"]
    if not sub_b.empty:
        q_x_b = sub_b["q_x"].values
        models_2026 = [
            ("Benchmark\nPromedio", sub_b["q_prom"]),
            ("EGARCH(1,3)\n_GED", sub_b["q_ged"]),
            (f"{egarch_x_name}\n(Ref. Econom.)", sub_b["q_x"]),
            ("CNN 1D\n(Hybrid 2+2)", sub_b["q_cnn"]),
            ("GRU\n(Hybrid 2+2)", sub_b["q_gru"]),
            ("LSTM\n(Hybrid 2+2)", sub_b["q_lstm"]),
            ("DL SuperEns.\n(60 Modelos)", sub_b["q_super"]),
        ]

        fig, ax = plt.subplots(figsize=(12.2, 5.7))
        x26 = np.arange(len(models_2026))
        w26 = 0.34
        means_26 = [float(m[1].mean()) for m in models_2026]
        meds_26 = [float(m[1].median()) for m in models_2026]
        winrates_26 = [
            np.nan if "(Ref. Econom.)" in m[0] else float(np.mean(m[1].values < q_x_b) * 100.0)
            for m in models_2026
        ]

        b_mean = ax.bar(x26 - w26 / 2, means_26, width=w26, color="#1f77b4", alpha=0.88, label="Mean QLIKE (2026)")
        b_med = ax.bar(x26 + w26 / 2, meds_26, width=w26, color="#2ca02c", alpha=0.88, label="Median QLIKE (2026)")

        for i, (bm, bmd, wr) in enumerate(zip(b_mean, b_med, winrates_26)):
            ax.text(bm.get_x() + bm.get_width() / 2.0, bm.get_height() + 0.03, f"{bm.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax.text(bmd.get_x() + bmd.get_width() / 2.0, bmd.get_height() + 0.03, f"{bmd.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            if np.isnan(wr):
                badge_txt = "Mejor Baseline\nEconométrico"
                badge_col = "#555555"
            else:
                badge_txt = f"Supera EGARCH-X\n{wr:.1f}% días"
                badge_col = "#1b7837" if wr > 50.0 else "#b2182b"

            ax.text(
                x26[i], max(bm.get_height(), bmd.get_height()) + 0.17,
                badge_txt,
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=badge_col,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=badge_col, alpha=0.94),
            )

        ax.set_ylim(0, max(means_26) * 1.31)
        ax.set_xticks(x26)
        ax.set_xticklabels([m[0] for m in models_2026], fontsize=9.5)
        ax.set_ylabel("Pérdida QLIKE (menor = mejor)")
        ax.set_title(f"Test Ciego 2026 (Fase B | T = {len(sub_b)}): Empate en Media vs. Porcentaje de Días Ganados sobre {egarch_x_name}")
        ax.legend(loc="upper right", frameon=True)
        _save_single_fig(fig, "09_comp_blind_test_2026_summary.png")


# ==============================================================================
# INSTANCIA 3: REGÍMENES DE ESTRÉS EXTREMO Y CRISIS COVID-19 (FIGURAS 10 A 12)
# ==============================================================================
def generate_diagnostics_individual_plots():
    print("\n[Instancia 3/3] Generando gráficos individuales de regímenes y estrés extremo (10 a 12)...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    reg_path = RESULTS_DIR / "regime_error_analysis.csv"
    shocks_path = RESULTS_DIR / "top15_extreme_shocks_models.csv"

    if not reg_path.exists() or not shocks_path.exists():
        raise FileNotFoundError("Ejecutá primero scripts/10_error_analysis_and_diagnostics.py.")

    df_reg = pd.read_csv(reg_path)
    df_shk = pd.read_csv(shocks_path)

    # 10. Reducción Porcentual Relativa (%) respecto a EGARCH(1,3)_GED por Régimen de Mercado
    target_regs = [
        "1_Calma (< P50)",
        "2_Normal (P50 - P90)",
        "3_Turbulencia_Alta (P90 - P98)",
        "4_Estres_Extremo (> P98)",
        "8_Test_Ciego_2026 (Fase B)",
    ]
    short_labels = ["Calma\n(<P50)", "Normal\n(P50–P90)", "Turbulencia Alta\n(P90–P98)", "Estrés Extremo\n(>P98)", "Test Ciego\n2026"]
    models_rel = [
        (egarch_x_name, "#e7298a"),
        ("CNN_1D_Hybrid_2x2", "#ff7f0e"),
        ("GRU_Hybrid_2x2", "#2ca02c"),
        ("LSTM_Hybrid_2x2", "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(12.2, 5.7))
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
                ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.6, "*", ha="center", va="bottom", fontsize=12, fontweight="bold")
            elif sig == "·":
                ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.6, "†", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", lw=1.1, label="Baseline de referencia: EGARCH(1,3)_GED (0%)")
    ax.plot([], [], " ", label="* Mejora significativa p < 0.05  |  † Mejora marginal p < 0.10 (Test DM unilateral)")
    ax.set_ylim(-9, 34)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Reducción porcentual de Mean QLIKE vs. EGARCH(1,3)_GED (%)")
    ax.set_title("Ventaja Condicional por Régimen de Volatilidad (Positivo = Menor Error que EGARCH_GED)")
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=9.2)
    _save_single_fig(fig, "10_diag_qlike_by_volatility_regime.png")

    # 11. Promedio de QLIKE y Peor Error Máximo en los Top-15 Shocks Históricos
    models_shk = [
        ("Benchmark Promedio", "QLIKE_Benchmark_Promedio", "#7f7f7f"),
        ("EGARCH(1,3)_GED", "QLIKE_EGARCH(1,3)_GED", "#d62728"),
        (egarch_x_name, f"QLIKE_{egarch_x_name}", "#e7298a"),
        ("CNN 1D (Hybrid)", "QLIKE_CNN_1D_Hybrid_2x2", "#ff7f0e"),
        ("GRU (Hybrid)", "QLIKE_GRU_Hybrid_2x2", "#2ca02c"),
        ("LSTM (Hybrid)", "QLIKE_LSTM_Hybrid_2x2", "#1f77b4"),
        ("DL SuperEns. (60M)", "QLIKE_DL_SuperEnsemble_60M", "#9467bd"),
    ]
    mean_top15 = [float(df_shk[m[1]].mean()) for m in models_shk]
    max_top15 = [float(df_shk[m[1]].max()) for m in models_shk]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.2, 5.6))
    xs = np.arange(len(models_shk))
    cols_s = [m[2] for m in models_shk]
    lbls_s = [m[0] for m in models_shk]

    b1 = ax_l.bar(xs, mean_top15, color=cols_s, alpha=0.88, edgecolor="black", linewidth=0.6)
    for bar in b1:
        hv = bar.get_height()
        ax_l.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.5, f"{hv:.2f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax_l.set_ylim(0, max(mean_top15) * 1.16)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(lbls_s, rotation=28, ha="right", fontsize=9.2)
    ax_l.set_ylabel("Promedio de QLIKE en los 15 shocks")
    ax_l.set_title("A. Error Promedio en los Top-15 Shocks\n(DL SuperEnsemble reduce -32.8% vs. EGARCH_GED)")

    b2 = ax_r.bar(xs, max_top15, color=cols_s, alpha=0.88, edgecolor="black", linewidth=0.6)
    for bar in b2:
        hv = bar.get_height()
        ax_r.text(bar.get_x() + bar.get_width() / 2.0, hv + 3.0, f"{hv:.1f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax_r.set_ylim(0, max(max_top15) * 1.16)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(lbls_s, rotation=28, ha="right", fontsize=9.2)
    ax_r.set_ylabel("Peor Pérdida Diaria (Max QLIKE)")
    ax_r.set_title("B. Contención de la Peor Explosión Individual\n(Black Thursday: 187.7 en EGARCH_GED vs. 125.8 en DL)")

    _save_single_fig(fig, "11_diag_top15_shocks_aggregate.png")

   # 12. Reacción Post-Shock en Crisis COVID-19 en escala %^2 (Varianza realizada suavizada 7d)
    c_covid = comp[(comp["Date"] >= "2020-02-01") & (comp["Date"] <= "2020-05-31")].copy()
    var_real_7d = (c_covid["y_true"] * 10000.0).rolling(7, min_periods=1).mean()
    s2_ged_pct2 = c_covid["s2_ged"] * 10000.0
    s2_x_pct2 = c_covid["s2_x"] * 10000.0
    s2_super_pct2 = c_covid["s2_super"] * 10000.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12.0, 5.8),
        gridspec_kw={"height_ratios": [1.0, 3.0], "hspace": 0.08}
    )

    for ax in (ax_top, ax_bot):
        ax.fill_between(
            c_covid["Date"], 0, var_real_7d,
            color="#bdbdbd", alpha=0.38, label="Varianza realizada móvil 7d (%²)"
        )
        ax.plot(c_covid["Date"], s2_ged_pct2, color="#d62728", lw=1.8, ls="--", label="EGARCH(1,3)_GED (sobrerreacción post-crash)")
        ax.plot(c_covid["Date"], s2_x_pct2, color="#e7298a", lw=1.8, ls="-.", label=egarch_x_name)
        ax.plot(c_covid["Date"], s2_super_pct2, color="#1f77b4", lw=2.3, label="DL SuperEnsemble (60M) (respuesta calibrada)")

    # Corte limpio sin pérdida de datos: muestra el pico de EGARCH (~1,540 %^2) y la media 7d (~355 %^2) arriba
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

    ax_top.set_title("Reacción Post-Shock en la Crisis COVID-19 (Febrero – Mayo 2020) en Escala %² (Eje Y Recortado)", pad=10)
    fig.supylabel(r"Varianza pronosticada $\hat{\sigma}^2_{t+1}$ (%$^2$)", x=0.015, fontsize=12)
    ax_top.legend(loc="upper right", frameon=True, ncol=2, fontsize=9.2)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))

    out_path_12 = OUT_DIR / "12_pred_covid_regime_response.png"
    fig.savefig(out_path_12, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path_12.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de los 12 Gráficos Individuales del Informe (300 DPI)")
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