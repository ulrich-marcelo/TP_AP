import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import PHASE_A_END_DATE
from src.evaluation import (
    compute_qlike_vector,
    diebold_mariano_full,
    get_significance_code,
)

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def _detect_egarch_x_name(df_base: pd.DataFrame) -> str:
    """Detecta dinámicamente el nombre del modelo EGARCH-X evaluado en baselines."""
    spec_path = RESULTS_DIR / "egarch_x_best_spec.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("spec") in df_base["model"].unique():
            return cfg["spec"]

    other_models = [
        m for m in df_base["model"].unique()
        if m not in ("Benchmark_Promedio", "EGARCH(1,3)_GED")
    ]
    if other_models:
        return str(other_models[0])
    raise ValueError("No se encontró ningún modelo EGARCH-X en baselines_predictions.parquet.")


def load_all_models_data():
    """
    Carga y alinea las predicciones out-of-sample de todos los modelos:
      1. Baselines: Benchmark_Promedio, EGARCH(1,3)_GED y EGARCH-X óptimo.
      2. Ensambles por Arquitectura: Hybrid_2x2 (principal), Mean_Top2 y Median_Top2.
      3. Candidatos individuales (Deep Ensemble de 5 semillas por config_id).
      4. Super-Ensamble global (promedio de las 3 arquitecturas Hybrid_2x2 = 60 modelos).
    """
    print("\n[1/4] Cargando series de predicciones fuera de muestra...")
    baselines_path = RESULTS_DIR / "baselines_predictions.parquet"
    cand_ens_path = RESULTS_DIR / "candidates_ensembles_predictions.parquet"
    arch_ens_path = RESULTS_DIR / "architecture_ensembles_predictions.parquet"

    for p in (baselines_path, cand_ens_path, arch_ens_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontró {p}. Ejecutá previamente los scripts 4 y 8.")

    df_base = pd.read_parquet(baselines_path)
    df_base["Date"] = pd.to_datetime(df_base["Date"])

    df_cand = pd.read_parquet(cand_ens_path)
    df_cand["Date"] = pd.to_datetime(df_cand["Date"])

    df_arch = pd.read_parquet(arch_ens_path)
    df_arch["Date"] = pd.to_datetime(df_arch["Date"])

    egarch_x_name = _detect_egarch_x_name(df_base)
    print(f"Modelo multivariado econométrico detectado: {egarch_x_name}")

    baselines = {}
    for m_name in ["Benchmark_Promedio", "EGARCH(1,3)_GED", egarch_x_name]:
        sub = df_base[df_base["model"] == m_name].sort_values("Date").reset_index(drop=True)
        baselines[m_name] = sub[["Date", "phase", "fold", "y_true", "sigma2_pred", "qlike"]].copy()

    dl_ensembles = {}
    for (arch, ens_type), sub in df_arch.groupby(["architecture", "ensemble_type"], sort=False):
        key = f"{arch}_{ens_type}"
        dl_ensembles[key] = (
            sub[["Date", "phase", "fold", "y_true", "sigma2_pred", "qlike"]]
            .sort_values("Date")
            .reset_index(drop=True)
        )

    # Super-Ensamble entre las 3 arquitecturas Hybrid_2x2 (60 modelos totales)
    sub_hybrids = df_arch[df_arch["ensemble_type"] == "Hybrid_2x2"]
    if not sub_hybrids.empty:
        super_ens = (
            sub_hybrids.groupby("Date", as_index=False)
            .agg(
                phase=("phase", "first"),
                fold=("fold", "first"),
                y_true=("y_true", "first"),
                sigma2_pred=("sigma2_pred", "mean"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
        super_ens["qlike"] = compute_qlike_vector(super_ens["y_true"].values, super_ens["sigma2_pred"].values)
        dl_ensembles["DL_SuperEnsemble_60M"] = super_ens

    dl_candidates = {}
    for cfg_id, sub in df_cand.groupby("config_id", sort=False):
        dl_candidates[cfg_id] = (
            pd.DataFrame({
                "Date": sub["Date"],
                "phase": sub["phase"],
                "fold": sub["fold"],
                "y_true": sub["y_true"],
                "sigma2_pred": sub["sigma2_pred_mean"],
                "qlike": sub["qlike_ensemble"],
            })
            .sort_values("Date")
            .reset_index(drop=True)
        )

    return baselines, dl_ensembles, dl_candidates, egarch_x_name


def run_comprehensive_dm_tests(baselines: dict, dl_ensembles: dict, dl_candidates: dict, egarch_x_name: str) -> pd.DataFrame:
    """
    Ejecuta contrastes Diebold-Mariano (con corrección HAC y HLN 1997) estratificados
    por Fase A (2019-2025), Fase B Test Ciego (2026) y Muestra Completa (2019-2026).
    """
    print("\n[2/4] Ejecutando contrastes estadísticos Diebold-Mariano (Fase A, Fase B Ciega y Muestra Completa)...")
    dm_records = []

    phase_scopes = ["Fase_A_Seleccion", "Fase_B_Test_Ciego", "Muestra_Completa"]

    def evaluate_pair(scope: str, base_name: str, cand_name: str, df_b: pd.DataFrame, df_c: pd.DataFrame, tipo_contraste: str):
        if scope != "Muestra_Completa":
            df_b = df_b[df_b["phase"] == scope]
            df_c = df_c[df_c["phase"] == scope]

        merged = pd.merge(
            df_b[["Date", "qlike"]].rename(columns={"qlike": "loss_base"}),
            df_c[["Date", "qlike"]].rename(columns={"qlike": "loss_cand"}),
            on="Date",
            how="inner",
        ).dropna()

        if len(merged) < 10:
            return None

        loss_b = merged["loss_base"].values
        loss_c = merged["loss_cand"].values
        dm_res = diebold_mariano_full(loss_b, loss_c, h=1)

        T = dm_res["T"]
        diff_mean = dm_res["diff_mean"]
        se_d = dm_res["se_hac"]
        dm_stat = dm_res["dm_stat"]
        p_val_two = dm_res["p_val_bilateral"]
        p_val_one = dm_res["p_val_unilateral"]

        sig_two = get_significance_code(p_val_two)
        sig_one = get_significance_code(p_val_one)

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
            "Fase": scope,
            "Tipo_Contraste": tipo_contraste,
            "Modelo_Base": base_name,
            "Modelo_Candidato": cand_name,
            "Observaciones_T": T,
            "QLIKE_Base": float(np.mean(loss_b)),
            "QLIKE_Candidato": float(np.mean(loss_c)),
            "Mediana_Base": float(np.median(loss_b)),
            "Mediana_Candidato": float(np.median(loss_c)),
            "Diferencia_Media": diff_mean,
            "Error_Estandar_HAC": se_d,
            "DM_Stat_HLN": dm_stat,
            "p_val_bilateral": p_val_two,
            "Signif_Bilateral": sig_two,
            "p_val_unilateral": p_val_one,
            "Signif_Unilateral": sig_one,
            "Veredicto_Bilateral": ver_two,
            "Veredicto_Unilateral": ver_one,
        }

    for scope in phase_scopes:
        # 1. Ensambles de Arquitectura DL vs. Baselines
        for base_name, base_df in baselines.items():
            for dl_name, dl_df in dl_ensembles.items():
                rec = evaluate_pair(scope, base_name, dl_name, base_df, dl_df, "DL_Ensemble vs Baseline")
                if rec is not None:
                    dm_records.append(rec)

        # 2. Candidatos Individuales DL (5 semillas) vs. Baselines
        for base_name, base_df in baselines.items():
            for cand_name, cand_df in dl_candidates.items():
                rec = evaluate_pair(scope, base_name, cand_name, base_df, cand_df, "DL_Candidato vs Baseline")
                if rec is not None:
                    dm_records.append(rec)

        # 3. Contrastes entre Baselines (incluye EGARCH_GED vs EGARCH-X)
        base_pairs = [
            ("Benchmark_Promedio", "EGARCH(1,3)_GED"),
            ("Benchmark_Promedio", egarch_x_name),
            ("EGARCH(1,3)_GED", egarch_x_name),
        ]
        for b1, b2 in base_pairs:
            rec = evaluate_pair(scope, b1, b2, baselines[b1], baselines[b2], "Inter-Baseline")
            if rec is not None:
                dm_records.append(rec)

        # 4. Contrastes entre Familias DL (Ensambles Hybrid_2x2)
        arch_pairs = [
            ("CNN_1D_Hybrid_2x2", "GRU_Hybrid_2x2"),
            ("CNN_1D_Hybrid_2x2", "LSTM_Hybrid_2x2"),
            ("GRU_Hybrid_2x2", "LSTM_Hybrid_2x2"),
        ]
        for a1, a2 in arch_pairs:
            if a1 in dl_ensembles and a2 in dl_ensembles:
                rec = evaluate_pair(scope, a1, a2, dl_ensembles[a1], dl_ensembles[a2], "Inter-Arquitectura DL")
                if rec is not None:
                    dm_records.append(rec)

        # 5. Ablación de Estrategia de Ensamble (Hybrid_2x2 vs Mean_Top2 y Median_Top2)
        for arch in ["CNN_1D", "GRU", "LSTM"]:
            hyb_key = f"{arch}_Hybrid_2x2"
            for alt_suffix in ["Mean_Top2", "Median_Top2"]:
                alt_key = f"{arch}_{alt_suffix}"
                if hyb_key in dl_ensembles and alt_key in dl_ensembles:
                    rec = evaluate_pair(
                        scope, alt_key, hyb_key, dl_ensembles[alt_key], dl_ensembles[hyb_key], "Ablacion Ensamble 2+2"
                    )
                    if rec is not None:
                        dm_records.append(rec)

    df_dm = pd.DataFrame(dm_records)
    df_dm.to_csv(RESULTS_DIR / "diebold_mariano_statistical_tests.csv", index=False)

    df_dm_blind = df_dm[df_dm["Fase"] == "Fase_B_Test_Ciego"].reset_index(drop=True)
    df_dm_blind.to_csv(RESULTS_DIR / "diebold_mariano_blind_test_2026.csv", index=False)

    return df_dm


def build_overall_comparison_summary(baselines: dict, dl_ensembles: dict, egarch_x_name: str) -> pd.DataFrame:
    """
    Genera una tabla resumen consolidada con métricas paramétricas, no paramétricas y
    tasa de victoria diaria (Win Rate) frente a EGARCH(1,3)_GED por fase.
    """
    all_models = {}
    for k, v in baselines.items():
        all_models[k] = ("Baseline", v)
    for k, v in dl_ensembles.items():
        if k.endswith("Hybrid_2x2") or k == "DL_SuperEnsemble_60M":
            all_models[k] = ("Deep_Learning", v)

    egarch_ref = baselines["EGARCH(1,3)_GED"][["Date", "phase", "qlike"]].rename(columns={"qlike": "qlike_egarch"})

    rows = []
    for scope in ["Fase_A_Seleccion", "Fase_B_Test_Ciego", "Muestra_Completa"]:
        for m_name, (m_cat, df_m) in all_models.items():
            sub = df_m if scope == "Muestra_Completa" else df_m[df_m["phase"] == scope]
            if sub.empty:
                continue

            merged = pd.merge(sub, egarch_ref[["Date", "qlike_egarch"]], on="Date", how="inner")
            win_rate = float(np.mean(merged["qlike"].values < merged["qlike_egarch"].values) * 100.0)

            rows.append({
                "Fase": scope,
                "Categoria": m_cat,
                "Modelo": m_name,
                "Observaciones": int(len(sub)),
                "Mean_QLIKE": float(sub["qlike"].mean()),
                "Median_QLIKE": float(sub["qlike"].median()),
                "Std_QLIKE": float(sub["qlike"].std(ddof=1)),
                "IQR_QLIKE": float(sub["qlike"].quantile(0.75) - sub["qlike"].quantile(0.25)),
                "Mean_Var_Pred_pct2": float(sub["sigma2_pred"].mean() * 10000.0),
                "WinRate_vs_EGARCH_pct": win_rate if m_name != "EGARCH(1,3)_GED" else 50.0,
            })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(RESULTS_DIR / "overall_models_comparison_summary.csv", index=False)
    return df_summary


def _shade_blind_test_2026(ax, max_date: pd.Timestamp):
    """Dibuja la frontera y sombreado de Fase B (Test Ciego 2026) si está presente en el eje."""
    cutoff = pd.Timestamp(PHASE_A_END_DATE)
    if max_date > cutoff:
        ax.axvline(cutoff, color="#4b0082", ls="--", lw=1.3, alpha=0.85)
        ax.axvspan(cutoff, max_date, color="#e6e6fa", alpha=0.35, label="Fase B: Test Ciego 2026")


def generate_comparative_plots(baselines: dict, dl_ensembles: dict, egarch_x_name: str):
    """Genera las figuras diagnósticas comparativas incluyendo el período ciego 2026."""
    print("\n[3/4] Generando visualizaciones diagnósticas comparativas...")

    prom_df = baselines["Benchmark_Promedio"]
    ged_df = baselines["EGARCH(1,3)_GED"]
    x_df = baselines[egarch_x_name]

    cnn_hyb = dl_ensembles["CNN_1D_Hybrid_2x2"]
    gru_hyb = dl_ensembles["GRU_Hybrid_2x2"]
    lstm_hyb = dl_ensembles["LSTM_Hybrid_2x2"]

    comp = pd.merge(
        prom_df[["Date", "phase", "fold", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_prom", "qlike": "q_prom"}),
        ged_df[["Date", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_ged", "qlike": "q_ged"}),
        on="Date",
    )
    comp = pd.merge(comp, x_df[["Date", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_x", "qlike": "q_x"}), on="Date")
    comp = pd.merge(comp, cnn_hyb[["Date", "y_true", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_cnn", "qlike": "q_cnn"}), on="Date")
    comp = pd.merge(comp, gru_hyb[["Date", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_gru", "qlike": "q_gru"}), on="Date")
    comp = pd.merge(comp, lstm_hyb[["Date", "sigma2_pred", "qlike"]].rename(columns={"sigma2_pred": "s2_lstm", "qlike": "q_lstm"}), on="Date")
    comp = comp.sort_values("Date").reset_index(drop=True)

    max_date = comp["Date"].max()

    # Seleccionar dinámicamente la mejor arquitectura DL según Fase A para el zoom detallado
    sub_a = comp[comp["phase"] == "Fase_A_Seleccion"]
    arch_means_a = {
        "CNN 1D (Hybrid 2+2)": ("s2_cnn", "q_cnn", float(sub_a["q_cnn"].mean())),
        "GRU (Hybrid 2+2)": ("s2_gru", "q_gru", float(sub_a["q_gru"].mean())),
        "LSTM (Hybrid 2+2)": ("s2_lstm", "q_lstm", float(sub_a["q_lstm"].mean())),
    }
    best_dl_label, (best_s2_col, best_q_col, _) = min(arch_means_a.items(), key=lambda kv: kv[1][2])

    # --------------------------------------------------------------------------
    # FIGURA 1: Pérdida Acumulada Diferencial (2019–2026)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.plot(comp["Date"], (comp["q_prom"] - comp[best_q_col]).cumsum(), label=f"Benchmark Promedio vs. {best_dl_label}", color="#7f7f7f", lw=1.8, ls=":")
    ax.plot(comp["Date"], (comp["q_ged"] - comp["q_cnn"]).cumsum(), label="EGARCH(1,3)_GED vs. CNN 1D (Hybrid 2+2)", color="#ff7f0e", lw=1.7, ls="--")
    ax.plot(comp["Date"], (comp["q_ged"] - comp["q_gru"]).cumsum(), label="EGARCH(1,3)_GED vs. GRU (Hybrid 2+2)", color="#2ca02c", lw=2.1)
    ax.plot(comp["Date"], (comp["q_ged"] - comp["q_lstm"]).cumsum(), label="EGARCH(1,3)_GED vs. LSTM (Hybrid 2+2)", color="#1f77b4", lw=2.1)
    ax.plot(comp["Date"], (comp["q_x"] - comp[best_q_col]).cumsum(), label=f"{egarch_x_name} vs. {best_dl_label}", color="#d62728", lw=1.8, ls="-.")

    _shade_blind_test_2026(ax, max_date)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"Diferencial Acumulado $\sum (QLIKE_{\mathrm{base}} - QLIKE_{\mathrm{DL}})$")
    ax.set_title("Evolución Temporal de la Superioridad Predictiva Out-of-Sample (2019–2026)\n(Pendiente Positiva = Deep Learning Supera al Baseline)")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cumulative_loss_differential.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURA 2: Trayectoria Completa (2019–2026) y Shock COVID-19
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.8), gridspec_kw={"height_ratios": [1.8, 1.2]})
    ax1.plot(comp["Date"], comp["y_true"] * 10000.0, color="#b0b0b0", alpha=0.5, lw=0.75, label=r"Proxy Real ($r_{t+1}^2 \times 10^4$)")
    ax1.plot(comp["Date"], comp["s2_prom"] * 10000.0, color="#7f7f7f", lw=1.1, ls=":", label="Benchmark Promedio")
    ax1.plot(comp["Date"], comp["s2_ged"] * 10000.0, color="#d62728", lw=1.25, label="EGARCH(1,3)_GED")
    ax1.plot(comp["Date"], comp[best_s2_col] * 10000.0, color="#1f77b4", lw=1.7, label=best_dl_label)
    _shade_blind_test_2026(ax1, max_date)
    ax1.set_ylabel(r"Varianza Porcentual $\hat{\sigma}^2_{t+1}$ (%$^2$)")
    ax1.set_title("Trayectoria de Pronóstico de Varianza de Bitcoin (Fase A: 2019–2025 | Fase B Ciega: 2026)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", frameon=True, ncol=2)

    mask_covid = (comp["Date"] >= "2020-02-01") & (comp["Date"] <= "2020-06-30")
    c_covid = comp[mask_covid]
    ax2.plot(c_covid["Date"], c_covid["y_true"] * 10000.0, color="black", alpha=0.45, lw=1.0, label="Proxy Real")
    ax2.plot(c_covid["Date"], c_covid["s2_ged"] * 10000.0, color="#d62728", lw=1.5, ls="--", label="EGARCH(1,3)_GED")
    ax2.plot(c_covid["Date"], c_covid["s2_x"] * 10000.0, color="#e7298a", lw=1.4, ls="-.", label=egarch_x_name)
    ax2.plot(c_covid["Date"], c_covid[best_s2_col] * 10000.0, color="#1f77b4", lw=2.0, label=best_dl_label)
    ax2.set_ylabel(r"$\hat{\sigma}^2_{t+1}$ (%$^2$)")
    ax2.set_title("Régimen de Estrés Extremo: Ruptura de Liquidez COVID-19 (Febrero – Junio 2020)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_variance_forecast_covid_regime.png")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURA 3: Distribución del Error QLIKE (Muestra Completa vs Test Ciego 2026)
    # --------------------------------------------------------------------------
    def _draw_boxplot_panel(ax, df_sub: pd.DataFrame, title: str):
        models_data = [
            ("Benchmark\nPromedio", df_sub["q_prom"], "#7f7f7f"),
            ("EGARCH(1,3)\nGED", df_sub["q_ged"], "#d95f02"),
            (f"{egarch_x_name}", df_sub["q_x"], "#e7298a"),
            ("CNN 1D\n(Hybrid 2+2)", df_sub["q_cnn"], "#ff7f0e"),
            ("GRU\n(Hybrid 2+2)", df_sub["q_gru"], "#2ca02c"),
            ("LSTM\n(Hybrid 2+2)", df_sub["q_lstm"], "#1f77b4"),
        ]
        labels = [m[0] for m in models_data]
        data = [m[1].values for m in models_data]
        colors = [m[2] for m in models_data]

        bplot = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            widths=0.45,
            meanprops={"marker": "D", "markerfacecolor": "gold", "markeredgecolor": "black", "markersize": 6.5},
            medianprops={"color": "black", "linewidth": 2.0},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
        )
        for patch, col in zip(bplot["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")

        for i, (_, s, col) in enumerate(models_data, start=1):
            mean_v = float(s.mean())
            med_v = float(s.median())
            q75 = float(s.quantile(0.75))
            y_pos = min(q75 + 0.42, 4.75)
            ax.text(
                i,
                y_pos,
                f"Med: {mean_v:.3f}\nMdn: {med_v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9),
            )
        ax.set_ylim(-0.2, 5.3)
        ax.set_ylabel(r"Pérdida Diaria $QLIKE_t$")
        ax.set_title(title)

    sub_b = comp[comp["phase"] == "Fase_B_Test_Ciego"]
    if not sub_b.empty:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6.2))
        _draw_boxplot_panel(ax_l, comp, f"A. Muestra Completa (2019–2026 | T = {len(comp)})")
        _draw_boxplot_panel(ax_r, sub_b, f"B. Test Ciego 2026 (Fase B | T = {len(sub_b)})")
        legend_elements = [
            Line2D([0], [0], color="black", lw=2.0, label="Línea central: Mediana"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=7.5, label="Rombo dorado: Media muestral"),
        ]
        ax_r.legend(handles=legend_elements, loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_qlike_error_distribution_boxplots.png")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        _draw_boxplot_panel(ax, comp, f"Distribución Out-of-Sample de Pérdidas QLIKE (T = {len(comp)})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_qlike_error_distribution_boxplots.png")
        plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURA 4: Desempeño Semestre a Semestre (Folds 0–13 Fase A + Folds 14–15 Ciego 2026)
    # --------------------------------------------------------------------------
    fold_summary = (
        comp.groupby(["fold", "phase"], as_index=False)
        .agg(
            q_prom=("q_prom", "mean"),
            q_ged=("q_ged", "mean"),
            q_x=("q_x", "mean"),
            q_best_dl=(best_q_col, "mean"),
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    n_folds = len(fold_summary)
    x_idx = np.arange(n_folds)
    w = 0.20

    ax.bar(x_idx - 1.5 * w, fold_summary["q_prom"], width=w, label="Benchmark Promedio", color="#7f7f7f", alpha=0.75)
    ax.bar(x_idx - 0.5 * w, fold_summary["q_ged"], width=w, label="EGARCH(1,3)_GED", color="#d62728", alpha=0.82)
    ax.bar(x_idx + 0.5 * w, fold_summary["q_x"], width=w, label=egarch_x_name, color="#e7298a", alpha=0.82)
    ax.bar(x_idx + 1.5 * w, fold_summary["q_best_dl"], width=w, label=best_dl_label, color="#1f77b4", alpha=0.90)

    blind_mask = fold_summary["phase"] == "Fase_B_Test_Ciego"
    if blind_mask.any():
        first_blind = int(fold_summary.loc[blind_mask, "fold"].index[0])
        ax.axvline(first_blind - 0.5, color="#4b0082", ls="--", lw=1.4)
        ax.axvspan(first_blind - 0.5, n_folds - 0.5, color="#e6e6fa", alpha=0.35, label="Fase B: Test Ciego 2026")

    tick_labels = [
        f"F{int(r['fold'])}\n({'2026*' if r['phase']=='Fase_B_Test_Ciego' else 'A'})"
        for _, r in fold_summary.iterrows()
    ]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("QLIKE Medio del Fold")
    ax.set_title("Consistencia Out-of-Sample Semestre a Semestre (Folds 0–13 Selección | Folds 14–15 Test Ciego 2026)")
    ax.legend(loc="upper right", frameon=True, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_fold_by_fold_comparison.png")
    plt.close(fig)

    print(f"Figuras guardadas exitosamente en: {FIG_DIR.resolve()}")


def main():
    baselines, dl_ensembles, dl_candidates, egarch_x_name = load_all_models_data()
    df_dm = run_comprehensive_dm_tests(baselines, dl_ensembles, dl_candidates, egarch_x_name)
    df_summary = build_overall_comparison_summary(baselines, dl_ensembles, egarch_x_name)
    generate_comparative_plots(baselines, dl_ensembles, egarch_x_name)

    print("\n" + "=" * 125)
    print("RESUMEN COMPARATIVO GENERAL POR FASE (GUARDADO EN results/overall_models_comparison_summary.csv):")
    print("=" * 125)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 125)
    print("CONTRASTES DIEBOLD-MARIANO (ENSAMBLES HYBRID 2+2 VS BASELINES):")
    print("=" * 125)
    sub_dm = df_dm[
        (df_dm["Tipo_Contraste"].isin(["DL_Ensemble vs Baseline", "Inter-Baseline"]))
        & (df_dm["Modelo_Candidato"].str.contains("Hybrid_2x2|SuperEnsemble|EGARCH"))
    ]
    disp_cols = [
        "Fase", "Modelo_Base", "Modelo_Candidato", "QLIKE_Base", "QLIKE_Candidato",
        "Diferencia_Media", "DM_Stat_HLN", "p_val_bilateral", "p_val_unilateral", "Signif_Unilateral",
    ]
    print(sub_dm[disp_cols].to_string(index=False))

    print("\nArtefactos exportados:")
    print(f"  - {RESULTS_DIR / 'diebold_mariano_statistical_tests.csv'}")
    print(f"  - {RESULTS_DIR / 'diebold_mariano_blind_test_2026.csv'}")
    print(f"  - {RESULTS_DIR / 'overall_models_comparison_summary.csv'}")


if __name__ == "__main__":
    main()