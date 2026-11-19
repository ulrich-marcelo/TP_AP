import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
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

# Configuración global de tipografía grande para máxima legibilidad en informes
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11.5,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
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


# ==============================================================================
# INSTANCIA 1: GRÁFICOS INDIVIDUALES DEL EDA (FASE A: 2016–2025)
# ==============================================================================
def generate_eda_individual_plots():
    print("\n[Instancia 1/4] Generando gráficos individuales de EDA...")
    from matplotlib.patches import Patch

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
    ax.bar(lags, acf_abs, width=w, color="#1f77b4", label=r"Magnitud absoluta $|r_t|$ (persistencia larga)")
    ax.bar(lags + w, acf_r2, width=w, color="#d62728", label=r"Varianza realizada $r_t^2$ (efecto ARCH)")
    ax.axhline(conf_band, color="black", ls="--", lw=1.1, label="Umbral ruido blanco (95%)")
    ax.axhline(-conf_band, color="black", ls="--", lw=1.1)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Rezago (días)")
    ax.set_ylabel("Autocorrelación (ACF)")
    ax.set_title("Memoria Temporal: Retornos Direccionales (Ruido Blanco) vs. Magnitud de Volatilidad")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "02_eda_acf_returns_vs_variance.png")

    # 03. News Impact Curve empírica por deciles (escala diaria directa + leyenda de colores)
    df["BTC_Decil"] = pd.qcut(df["BTC_r_oc"], q=10, labels=[f"D{d}" for d in range(1, 11)])
    nic = df.groupby("BTC_Decil", observed=False).agg(
        BTC_r_oc_Medio_pct=("BTC_r_oc", lambda s: float(s.mean() * 100.0)),
        Target_r2_Media_pct2=("target_r2_t1", lambda s: float(s.mean() * 10000.0)),
        Vol_Diaria_pct=("target_r2_t1", lambda s: float(np.sqrt(s.mean()) * 100.0)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x_pos = np.arange(len(nic))
    bar_colors = ["#d62728" if m < -2.0 else ("#ff7f0e" if m > 2.0 else "#1f77b4") for m in nic["BTC_r_oc_Medio_pct"]]
    bars = ax.bar(x_pos, nic["Target_r2_Media_pct2"], color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    for bar, (_, row) in zip(bars, nic.iterrows()):
        h_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val + 0.45,
            f"{h_val:.1f} %²\n({row['Vol_Diaria_pct']:.2f}%/d)",
            ha="center", va="bottom", fontsize=8.6, fontweight="bold",
        )

    legend_patches = [
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.85, label="Caídas fuertes (retorno medio < -2%)"),
        Patch(facecolor="#1f77b4", edgecolor="black", alpha=0.85, label="Movimientos moderados (-2% a +2%)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", alpha=0.85, label="Subas fuertes (retorno medio > +2%)"),
    ]
    ax.legend(handles=legend_patches, loc="upper center", frameon=True)
    ax.set_ylim(0, nic["Target_r2_Media_pct2"].max() * 1.25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}\n({m:+.1f}%)" for d, m in zip(nic["BTC_Decil"], nic["BTC_r_oc_Medio_pct"])])
    ax.set_xlabel("Decil de retorno diario BTC (retorno medio del decil)")
    ax.set_ylabel(r"Varianza media al día siguiente $\mathbb{E}[r_{t+1}^2]$ (%$^2$)")
    ax.set_title("News Impact Curve Empírica: Efecto Apalancamiento Asimétrico (Varianza %² y Desvío Diario)")
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

    # 05. Línea temporal de Outliers Extremos con eje Y recortado (Broken Axis)
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

    # Límites sin perder ninguna observación intermedia (entre 507 y 2160 no hay puntos)
    ax_top.set_ylim(1850, 2320)
    ax_bot.set_ylim(-15, 585)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    # Marcas diagonales de corte de eje
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d * 3.3, +d * 3.3), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 3.3, +d * 3.3), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    letter_offsets = [(-22, 14), (8, 28), (0, 18), (18, 16), (20, 0)]
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

    # 06. QQ-Plot anotado de colas pesadas (sin solapamiento con el título)
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
# INSTANCIA 2: COMPARACIÓN GENERAL Y TEST CIEGO 2026 (DL VS BASELINES)
# ==============================================================================
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


def generate_comparison_individual_plots():
    print("\n[Instancia 2/4] Generando gráficos individuales comparativos (DL vs. Baselines)...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    max_date = comp["Date"].max()

    # 07. Diferencial Acumulado de Pérdida QLIKE: DL_SuperEnsemble_60M vs. los 3 Baselines (2019-2026)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.plot(
        comp["Date"], (comp["q_prom"] - comp["q_super"]).cumsum(),
        label="Benchmark Promedio vs. DL SuperEnsemble (60M)", color="#7f7f7f", lw=1.9, ls=":"
    )
    ax.plot(
        comp["Date"], (comp["q_ged"] - comp["q_super"]).cumsum(),
        label="EGARCH(1,3)_GED vs. DL SuperEnsemble (60M)", color="#d62728", lw=2.2, ls="-"
    )
    ax.plot(
        comp["Date"], (comp["q_x"] - comp["q_super"]).cumsum(),
        label=f"{egarch_x_name} vs. DL SuperEnsemble (60M)", color="#1f77b4", lw=2.3, ls="-"
    )
    ax.plot(
        comp["Date"], (comp["q_x"] - comp["q_lstm"]).cumsum(),
        label=f"{egarch_x_name} vs. LSTM (Hybrid 2+2)", color="#2ca02c", lw=1.7, ls="--"
    )
    _shade_blind_test_2026(ax, max_date)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"Diferencial Acumulado $\sum (QLIKE_{\mathrm{base}} - QLIKE_{\mathrm{DL}})$")
    ax.set_title("Superioridad Predictiva Acumulada Out-of-Sample: SuperEnsemble DL vs. Baselines (2019–2026)\n(Pendiente Positiva = Deep Learning Supera al Baseline)")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _save_single_fig(fig, "07_comp_cumulative_loss_differential.png")

    # 08. Desempeño Fold a Fold con Año y Semestre: DL_SuperEnsemble_60M vs. los 3 Baselines
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
        return f"F{int(row['fold'])}\n{mid_dt.year}-{sem}{suffix}"

    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    x_idx = np.arange(len(fold_sum))
    w = 0.20
    ax.bar(x_idx - 1.5 * w, fold_sum["q_prom"], width=w, label="Benchmark Promedio", color="#7f7f7f", alpha=0.78)
    ax.bar(x_idx - 0.5 * w, fold_sum["q_ged"], width=w, label="EGARCH(1,3)_GED", color="#d62728", alpha=0.85)
    ax.bar(x_idx + 0.5 * w, fold_sum["q_x"], width=w, label=egarch_x_name, color="#e7298a", alpha=0.85)
    ax.bar(x_idx + 1.5 * w, fold_sum["q_super"], width=w, label="DL SuperEnsemble (60M)", color="#1f77b4", alpha=0.92)

    blind_mask = fold_sum["phase"] == "Fase_B_Test_Ciego"
    if blind_mask.any():
        first_blind = int(fold_sum.loc[blind_mask].index[0])
        ax.axvline(first_blind - 0.5, color="#4b0082", ls="--", lw=1.5)
        ax.axvspan(first_blind - 0.5, len(fold_sum) - 0.5, color="#e6e6fa", alpha=0.38, label="Fase B: Test Ciego 2026 (*)")

    ax.set_xticks(x_idx)
    ax.set_xticklabels([_format_fold_label(r) for _, r in fold_sum.iterrows()], fontsize=9.2)
    ax.set_ylabel("QLIKE Medio del Semestre")
    ax.set_title("Consistencia Semestre a Semestre (2019–2026): DL SuperEnsemble vs. Baselines Econométricos")
    ax.legend(loc="upper right", frameon=True, ncol=3)
    _save_single_fig(fig, "08_comp_fold_by_fold_qlike.png")

    # 09. Comportamiento en Outliers y Cola Superior (Percentiles P90, P95, P99 y Std_QLIKE en Muestra Completa)
    models_tail = [
        ("Benchmark\nPromedio", comp["q_prom"]),
        ("EGARCH(1,3)\nGED", comp["q_ged"]),
        (egarch_x_name, comp["q_x"]),
        ("CNN 1D\n(Hybrid 2+2)", comp["q_cnn"]),
        ("GRU\n(Hybrid 2+2)", comp["q_gru"]),
        ("LSTM\n(Hybrid 2+2)", comp["q_lstm"]),
        ("DL SuperEns.\n(60 Modelos)", comp["q_super"]),
    ]
    labels_t = [m[0] for m in models_tail]
    p90_vals = [float(np.percentile(m[1], 90)) for m in models_tail]
    p95_vals = [float(np.percentile(m[1], 95)) for m in models_tail]
    p99_vals = [float(np.percentile(m[1], 99)) for m in models_tail]
    std_vals = [float(np.std(m[1], ddof=1)) for m in models_tail]

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    xt = np.arange(len(models_tail))
    wt = 0.20
    ax.bar(xt - 1.5 * wt, p90_vals, width=wt, color="#6baed6", label="Percentil 90 QLIKE")
    ax.bar(xt - 0.5 * wt, p95_vals, width=wt, color="#2171b5", label="Percentil 95 QLIKE")
    bars_p99 = ax.bar(xt + 0.5 * wt, p99_vals, width=wt, color="#d62728", alpha=0.88, label="Percentil 99 QLIKE (Peor 1% de días)")
    bars_std = ax.bar(xt + 1.5 * wt, std_vals, width=wt, color="#ff7f0e", alpha=0.88, label="Desvío Estándar Diario (Std QLIKE)")

    for bar in bars_p99:
        hv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.25, f"{hv:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#8c1515")
    for bar in bars_std:
        hv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, hv + 0.25, f"{hv:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#8c4500")

    ax.set_ylim(0, max(max(p99_vals), max(std_vals)) * 1.18)
    ax.set_xticks(xt)
    ax.set_xticklabels(labels_t)
    ax.set_ylabel("Pérdida QLIKE / Dispersión en Cola Superior")
    ax.set_title(f"Comportamiento ante Outliers (Muestra Completa 2019–2026 | T = {len(comp)}): Cuantiles Extremos y Dispersión")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "09_comp_upper_tail_loss_quantiles.png")

    # 10. Test Ciego 2026: Media vs. Mediana QLIKE y Tasa de Victoria Diaria (Win-Rate vs EGARCH_GED)
    sub_b = comp[comp["phase"] == "Fase_B_Test_Ciego"]
    if not sub_b.empty:
        q_ged_b = sub_b["q_ged"].values
        models_2026 = [
            ("Benchmark\nPromedio", sub_b["q_prom"], "#7f7f7f"),
            ("EGARCH(1,3)\nGED (Ref.)", sub_b["q_ged"], "#d95f02"),
            (egarch_x_name, sub_b["q_x"], "#e7298a"),
            ("CNN 1D\n(Hybrid 2+2)", sub_b["q_cnn"], "#ff7f0e"),
            ("GRU\n(Hybrid 2+2)", sub_b["q_gru"], "#2ca02c"),
            ("LSTM\n(Hybrid 2+2)", sub_b["q_lstm"], "#1f77b4"),
            ("DL SuperEns.\n(60 Modelos)", sub_b["q_super"], "#9467bd"),
        ]

        fig, ax = plt.subplots(figsize=(12.0, 5.6))
        x26 = np.arange(len(models_2026))
        w26 = 0.34
        means_26 = [float(m[1].mean()) for m in models_2026]
        meds_26 = [float(m[1].median()) for m in models_2026]
        winrates_26 = [
            50.0 if "Ref." in m[0] else float(np.mean(m[1].values < q_ged_b) * 100.0)
            for m in models_2026
        ]

        b_mean = ax.bar(x26 - w26 / 2, means_26, width=w26, color="#1f77b4", alpha=0.88, label="Mean QLIKE (2026)")
        b_med = ax.bar(x26 + w26 / 2, meds_26, width=w26, color="#2ca02c", alpha=0.88, label="Median QLIKE (2026)")

        for i, (bm, bmd, wr) in enumerate(zip(b_mean, b_med, winrates_26)):
            ax.text(bm.get_x() + bm.get_width() / 2.0, bm.get_height() + 0.03, f"{bm.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax.text(bmd.get_x() + bmd.get_width() / 2.0, bmd.get_height() + 0.03, f"{bmd.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            badge_col = "#1b7837" if wr > 50.0 else ("#555555" if wr == 50.0 else "#b2182b")
            ax.text(
                x26[i], max(bm.get_height(), bmd.get_height()) + 0.17,
                f"Win: {wr:.1f}%",
                ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=badge_col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=badge_col, alpha=0.92),
            )

        ax.set_ylim(0, max(means_26) * 1.24)
        ax.set_xticks(x26)
        ax.set_xticklabels([m[0] for m in models_2026])
        ax.set_ylabel("Pérdida QLIKE (menor = mejor)")
        ax.set_title(f"Test Ciego 2026 (Fase B | T = {len(sub_b)}): Empate en Media vs. Dominio Diario (Win-Rate y Mediana) de Recurrentes")
        ax.legend(loc="upper right", frameon=True)
        _save_single_fig(fig, "10_comp_blind_test_2026_winrate_and_loss.png")


def generate_diagnostics_individual_plots():
    print("\n[Instancia 3/4] Generando gráficos individuales de regímenes y diagnóstico...")
    comp, egarch_x_name = _load_merged_oos_dataframe()
    reg_path = RESULTS_DIR / "regime_error_analysis.csv"
    shocks_path = RESULTS_DIR / "top15_extreme_shocks_models.csv"

    if not reg_path.exists() or not shocks_path.exists():
        raise FileNotFoundError("Ejecutá primero scripts/10_error_analysis_and_diagnostics.py.")

    df_reg = pd.read_csv(reg_path)
    df_shk = pd.read_csv(shocks_path)

    # 11. Mejora Porcentual Relativa (%) respecto a EGARCH(1,3)_GED por Régimen de Mercado
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
        ("DL_SuperEnsemble_60M", "#9467bd"),
    ]

    fig, ax = plt.subplots(figsize=(12.2, 5.7))
    x = np.arange(len(target_regs))
    w = 0.16

    for idx, (m_name, col) in enumerate(models_rel):
        imp_vals = []
        sigs = []
        for reg in target_regs:
            ged_val = float(df_reg[(df_reg["Regimen"] == reg) & (df_reg["Modelo"] == "EGARCH(1,3)_GED")]["Mean_QLIKE"].iloc[0])
            sub_m = df_reg[(df_reg["Regimen"] == reg) & (df_reg["Modelo"] == m_name)].iloc[0]
            m_val = float(sub_m["Mean_QLIKE"])
            imp_pct = (ged_val - m_val) / ged_val * 100.0
            imp_vals.append(imp_pct)
            sigs.append(str(sub_m["Sig_vs_GED"]))

        offset = (idx - 2) * w
        bars = ax.bar(x + offset, imp_vals, width=w, label=m_name, color=col, alpha=0.88)
        for bar, sig in zip(bars, sigs):
            if sig in ("***", "**", "*", "·"):
                hv = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    hv + 0.6 if hv >= 0 else hv - 1.6,
                    sig,
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="black",
                )

    ax.axhline(0, color="black", lw=1.1, label="Referencia: EGARCH(1,3)_GED (0%)")
    ax.set_ylim(-9, 33)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel("Reducción porcentual de Mean QLIKE vs. EGARCH(1,3)_GED (%)")
    ax.set_title("Ganancia Relativa por Régimen de Volatilidad (Positivo = Supera a EGARCH_GED | *** p<0.001, * p<0.05, · p<0.10)")
    ax.legend(loc="upper left", frameon=True, ncol=3, fontsize=9.2)
    _save_single_fig(fig, "11_diag_qlike_by_volatility_regime.png")

    # 12. Top-8 Shocks Históricos: DL_SuperEnsemble_60M vs. los 3 Baselines (Eje Y Recortado)
    top8 = df_shk.head(8).copy()
    x8 = np.arange(len(top8))
    w8 = 0.20

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(12.5, 6.0),
        gridspec_kw={"height_ratios": [1.1, 2.8], "hspace": 0.08}
    )

    series_shocks = [
        ("QLIKE_Benchmark_Promedio", "Benchmark Promedio", "#7f7f7f", -1.5 * w8),
        ("QLIKE_EGARCH(1,3)_GED", "EGARCH(1,3)_GED", "#d62728", -0.5 * w8),
        (f"QLIKE_{egarch_x_name}", egarch_x_name, "#e7298a", +0.5 * w8),
        ("QLIKE_DL_SuperEnsemble_60M", "DL SuperEnsemble (60M)", "#1f77b4", +1.5 * w8),
    ]

    for ax in (ax_top, ax_bot):
        for col_name, lbl, col_hex, off in series_shocks:
            bars = ax.bar(x8 + off, top8[col_name], width=w8, color=col_hex, label=lbl, alpha=0.90)
            for bar in bars:
                hv = bar.get_height()
                if (ax is ax_top and hv > 80) or (ax is ax_bot and 18 <= hv <= 40):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        hv + (2.5 if ax is ax_top else 0.7),
                        f"{hv:.0f}",
                        ha="center", va="bottom", fontsize=8.0, fontweight="bold",
                    )

    # Entre 35 y 91 no hay ninguna observación: corte limpio sin pérdida de datos
    ax_top.set_ylim(82, 208)
    ax_bot.set_ylim(0, 40)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.011
    kw = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d * 2.8, +d * 2.8), **kw)
    ax_top.plot((1 - d, 1 + d), (-d * 2.8, +d * 2.8), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_bot.set_xticks(x8)
    ax_bot.set_xticklabels([f"{r['Event_Date_t1']}\n({r['BTC_Ret_t1_pct']:+.1f}%)" for _, r in top8.iterrows()], fontsize=9.2)
    ax_top.set_title("Pérdidas Puntuales en los Top-8 Mayores Shocks de Bitcoin: DL SuperEnsemble vs. Baselines (Eje Recortado)", pad=10)
    fig.supylabel("Pérdida Diaria QLIKE en el Evento", x=0.015, fontsize=12)
    ax_top.legend(loc="upper right", frameon=True, ncol=2, fontsize=9.2)

    out_path_12 = OUT_DIR / "12_diag_top8_extreme_shocks_qlike.png"
    fig.savefig(out_path_12, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path_12.name}")

    # 13. Zoom Crisis COVID-19 (Feb - Jun 2020)
    c_covid = comp[(comp["Date"] >= "2020-02-01") & (comp["Date"] <= "2020-06-30")]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(c_covid["Date"], c_covid["y_true"] * 10000.0, color="black", alpha=0.45, lw=1.1, label=r"Proxy Realizada $r_{t+1}^2$ (%$^2$)")
    ax.plot(c_covid["Date"], c_covid["s2_ged"] * 10000.0, color="#d62728", lw=1.6, ls="--", label="EGARCH(1,3)_GED")
    ax.plot(c_covid["Date"], c_covid["s2_x"] * 10000.0, color="#e7298a", lw=1.6, ls="-.", label=egarch_x_name)
    ax.plot(c_covid["Date"], c_covid["s2_super"] * 10000.0, color="#1f77b4", lw=2.1, label="DL SuperEnsemble (60M)")
    ax.set_yscale("log")
    ax.set_ylabel(r"Varianza $\hat{\sigma}^2_{t+1}$ (%$^2$, escala log)")
    ax.set_title("Dinámica de Pronóstico durante la Ruptura de Liquidez COVID-19 (Febrero – Junio 2020)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", frameon=True)
    _save_single_fig(fig, "13_diag_covid_crisis_trajectory.png")

    # 14. Correlograma ACF de residuos estandarizados al cuadrado z_{t+1}^2
    df_ref = load_returns_reference(DATA_PATH, phase="all")
    df_proc, _, _, _, _ = load_processed_data_with_dates(DATA_PATH, phase="all")
    r_cc_t1 = df_ref["BTC_r_cc"].shift(-1).fillna(
        np.sign(df_proc["BTC_r_oc"].shift(-1)).fillna(1.0) * np.sqrt(df_proc["target_r2_t1"])
    )
    r_map = pd.DataFrame({"Date": pd.to_datetime(df_proc.index), "r_t1": r_cc_t1.values})
    comp_r = pd.merge(comp, r_map, on="Date", how="inner")

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    max_lag = 25
    lags = np.arange(1, max_lag + 1)
    conf_band = 1.96 / np.sqrt(len(comp_r))

    for col_s2, label, color, ls in [
        ("s2_prom", "Benchmark Promedio", "#7f7f7f", ":"),
        ("s2_ged", "EGARCH(1,3)_GED", "#d62728", "--"),
        ("s2_x", egarch_x_name, "#e7298a", "-."),
        ("s2_gru", "GRU (Hybrid 2+2)", "#2ca02c", "-"),
        ("s2_lstm", "LSTM (Hybrid 2+2)", "#1f77b4", "-"),
        ("s2_super", "DL SuperEnsemble (60M)", "#9467bd", "-"),
    ]:
        z = comp_r["r_t1"].values / np.sqrt(np.maximum(comp_r[col_s2].values, 1e-12))
        acf_vals = acf(z ** 2, nlags=max_lag, fft=True)[1:]
        ax.plot(lags, acf_vals, marker="o", markersize=4.2, lw=1.6, ls=ls, color=color, label=f"{label} ($z^2$)")

    ax.axhline(conf_band, color="black", ls="--", lw=1.0, label="Banda ruido blanco (95%)")
    ax.axhline(-conf_band, color="black", ls="--", lw=1.0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Rezago (días)")
    ax.set_ylabel(r"Autocorrelación ACF($z_{t+1}^2$)")
    ax.set_title("Diagnóstico de Calibración: Autocorrelación Remanente en Residuos Estandarizados al Cuadrado")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "14_diag_standardized_residuals_acf.png")
# ==============================================================================
# INSTANCIA 4: TRAYECTORIAS E INCERTIDUMBRE EPISTÉMICA DE LOS ENSAMBLES
# ==============================================================================
def generate_prediction_individual_plots():
    print("\n[Instancia 4/4] Generando gráficos individuales de trayectoria e incertidumbre...")
    comp, egarch_x_name = _load_merged_oos_dataframe()

    # 15. Test Ciego 2026 con banda de incertidumbre epistémica (+/- 1 std del ensamble)
    sub_b = comp[comp["phase"] == "Fase_B_Test_Ciego"].copy()
    if not sub_b.empty:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        y_real = sub_b["y_true"] * 10000.0
        s2_gru = sub_b["s2_gru"] * 10000.0
        std_gru = sub_b["std_gru"] * 10000.0
        s2_x = sub_b["s2_x"] * 10000.0

        ax.plot(sub_b["Date"], y_real, color="#9e9e9e", alpha=0.55, lw=1.0, label=r"Varianza Realizada $r_{t+1}^2$ (%$^2$)")
        ax.plot(sub_b["Date"], s2_x, color="#e7298a", lw=1.5, ls="--", label=egarch_x_name)
        ax.plot(sub_b["Date"], s2_gru, color="#2ca02c", lw=2.0, label="GRU Hybrid 2+2 (Media 20 modelos)")
        ax.fill_between(
            sub_b["Date"],
            np.maximum(s2_gru - std_gru, 0.1),
            s2_gru + std_gru,
            color="#2ca02c",
            alpha=0.22,
            label=r"Incertidumbre epistémica ($\pm 1\sigma$ entre modelos)",
        )
        ax.set_yscale("log")
        ax.set_ylabel(r"Varianza Pronosticada (%$^2$, escala log)")
        ax.set_title("Test Ciego 2026 (Fase B): Trayectoria de Pronóstico e Incertidumbre Epistémica del Ensamble")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(loc="upper right", frameon=True, ncol=2)
        _save_single_fig(fig, "15_pred_blind_test_2026_uncertainty.png")

    # 16. Trayectoria completa 2019-2026 en escala logarítmica
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.plot(comp["Date"], comp["y_true"] * 10000.0, color="#b0b0b0", alpha=0.5, lw=0.8, label=r"Varianza Realizada $r_{t+1}^2$ (%$^2$)")
    ax.plot(comp["Date"], comp["s2_x"] * 10000.0, color="#e7298a", lw=1.25, ls="--", label=egarch_x_name)
    ax.plot(comp["Date"], comp["s2_lstm"] * 10000.0, color="#1f77b4", lw=1.6, label="LSTM (Hybrid 2+2)")
    _shade_blind_test_2026(ax, comp["Date"].max())
    ax.set_yscale("log")
    ax.set_ylabel(r"Varianza $\hat{\sigma}^2_{t+1}$ (%$^2$, escala log)")
    ax.set_title("Serie Completa Out-of-Sample (2019–2026): Pronóstico de Volatilidad en Fase A y Fase B Ciega")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", frameon=True, ncol=2)
    _save_single_fig(fig, "16_pred_full_timeline_log_scale.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Gráficos Individuales para el Informe (300 DPI)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "eda", "comparison", "diagnostics", "predictions"],
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
    if args.stage in ("all", "predictions"):
        generate_prediction_individual_plots()

    print("\nGeneración de gráficos individuales finalizada.")