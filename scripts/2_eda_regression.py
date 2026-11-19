import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, acf, kpss

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

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
FIG_DIR = RESULTS_DIR / "figures"
EDA_DIR.mkdir(parents=True, exist_ok=True)
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


# ==============================================================================
# BLOQUE 1: ESTACIONARIEDAD, COLAS PESADAS Y VOLATILITY CLUSTERING
# ==============================================================================
def analyze_stationarity_and_tails(df_full: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Evalúa estacionariedad (ADF y KPSS), momentos de orden superior (asimetría,
    exceso de curtosis) y normalidad (Jarque-Bera) sobre features, retornos y target.
    """
    cols_to_audit = feature_names + ["BTC_r_cc", "target_r2_t1"]
    rows = []

    for col in cols_to_audit:
        s = df_full[col].dropna().values.astype(np.float64)

        adf_res = adfuller(s, autolag="AIC")
        adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])

        try:
            kpss_res = kpss(s, regression="c", nlags="auto")
            kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

        jb_stat, jb_p = stats.jarque_bera(s)

        rows.append({
            "Variable": col,
            "Observaciones": len(s),
            "Media": float(np.mean(s)),
            "Std": float(np.std(s, ddof=1)),
            "Mediana": float(np.median(s)),
            "IQR": float(np.percentile(s, 75) - np.percentile(s, 25)),
            "Min": float(np.min(s)),
            "Max": float(np.max(s)),
            "Asimetria": float(stats.skew(s)),
            "Exceso_Curtosis": float(stats.kurtosis(s, fisher=True)),
            "Jarque_Bera_Stat": float(jb_stat),
            "Jarque_Bera_p": float(jb_p),
            "ADF_Stat": adf_stat,
            "ADF_p_val": adf_p,
            "KPSS_Stat": kpss_stat,
            "KPSS_p_val": kpss_p,
            "Sin_Raiz_Unitaria_ADF": adf_p < 0.05,
            "Regimen_Estable_KPSS": np.isnan(kpss_p) or (kpss_p > 0.01),
        })

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(EDA_DIR / "eda_stationarity_and_tails.csv", index=False)
    return df_stats


def analyze_volatility_clustering(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Contrasta formalmente la presencia de volatility clustering comparando la
    autocorrelación de retornos crudos r_t frente a |r_t| y r_t^2 (Ljung-Box y ARCH-LM).
    """
    r = df_full["BTC_r_cc"].values.astype(np.float64)
    abs_r = np.abs(r)
    r2 = r ** 2

    lags_to_test = [1, 5, 10, 20, 30]
    lb_r = acorr_ljungbox(r, lags=lags_to_test, return_df=True)
    lb_abs = acorr_ljungbox(abs_r, lags=lags_to_test, return_df=True)
    lb_r2 = acorr_ljungbox(r2, lags=lags_to_test, return_df=True)

    rows = []
    for lag in lags_to_test:
        arch_stat, arch_p, _, _ = het_arch(r, nlags=lag)
        rows.append({
            "Lag": lag,
            "LB_Retornos_r_Stat": float(lb_r.loc[lag, "lb_stat"]),
            "LB_Retornos_r_p": float(lb_r.loc[lag, "lb_pvalue"]),
            "LB_Abs_Retornos_Stat": float(lb_abs.loc[lag, "lb_stat"]),
            "LB_Abs_Retornos_p": float(lb_abs.loc[lag, "lb_pvalue"]),
            "LB_Varianza_r2_Stat": float(lb_r2.loc[lag, "lb_stat"]),
            "LB_Varianza_r2_p": float(lb_r2.loc[lag, "lb_pvalue"]),
            "Engle_ARCH_LM_Stat": float(arch_stat),
            "Engle_ARCH_LM_p": float(arch_p),
        })

    df_vc = pd.DataFrame(rows)
    df_vc.to_csv(EDA_DIR / "eda_volatility_clustering_tests.csv", index=False)
    return df_vc


def plot_volatility_clustering(df_full: pd.DataFrame, df_outliers: pd.DataFrame) -> None:
    """
    Figura 1: Evidencia visual de Volatility Clustering y persistencia de varianza.
    """
    dates = df_full.index
    r_pct = df_full["BTC_r_cc"] * 100.0
    r2_pct2 = (df_full["BTC_r_cc"] ** 2) * 10000.0
    rolling_vol_21d = r_pct.rolling(21).std()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), gridspec_kw={"height_ratios": [1.15, 1.15, 1.0]})

    # Panel A: Retornos diarios BTC
    ax1 = axes[0]
    ax1.plot(dates, r_pct, color="#1f77b4", lw=0.75, alpha=0.85, label=r"Retorno diario $r_{BTC, t}$ (%)")
    ax1.fill_between(dates, -2 * rolling_vol_21d, 2 * rolling_vol_21d, color="#ff7f0e", alpha=0.22,
                     label=r"Banda dinámica $\pm 2\hat{\sigma}_{21d}$")
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Retorno (%)")
    ax1.set_title("A. Retornos Logarítmicos Diarios de Bitcoin y Envolvente de Volatilidad Móvil (21d)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: Proxy de varianza r_t^2 y clústeres de volatilidad
    ax2 = axes[1]
    ax2.plot(dates, r2_pct2, color="#7f7f7f", lw=0.7, alpha=0.55, label=r"Varianza realizada diaria $r_t^2$ (%$^2$)")
    ax2.plot(dates, rolling_vol_21d ** 2, color="#d62728", lw=1.6, label=r"Varianza móvil 21 días $\hat{\sigma}^2_{21d}$ (%$^2$)")
    for _, row in df_outliers.head(5).iterrows():
        ev_date = pd.to_datetime(row["Event_Date_t1"])
        if ev_date in df_full.index:
            ax2.axvline(ev_date, color="#d62728", ls=":", lw=1.0, alpha=0.65)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Varianza (%$^2$, esc. log)")
    ax2.set_title("B. Clústeres de Volatilidad (Volatility Clustering): Alternancia de Regímenes de Calma y Estrés")
    ax2.legend(loc="upper right", frameon=True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel C: ACF comparativa r_t vs |r_t| vs r_t^2
    ax3 = axes[2]
    max_lag = 30
    lags = np.arange(1, max_lag + 1)
    acf_r = acf(df_full["BTC_r_cc"].values, nlags=max_lag, fft=True)[1:]
    acf_abs = acf(np.abs(df_full["BTC_r_cc"].values), nlags=max_lag, fft=True)[1:]
    acf_r2 = acf(df_full["BTC_r_cc"].values ** 2, nlags=max_lag, fft=True)[1:]
    conf_band = 1.96 / np.sqrt(len(df_full))

    w = 0.27
    ax3.bar(lags - w, acf_r, width=w, color="#7f7f7f", label=r"Retornos crudos $r_t$ (sin memoria lineal)")
    ax3.bar(lags, acf_abs, width=w, color="#1f77b4", label=r"Magnitud $|r_t|$ (persistencia fuerte)")
    ax3.bar(lags + w, acf_r2, width=w, color="#d62728", label=r"Varianza $r_t^2$ (efecto ARCH)")
    ax3.axhline(conf_band, color="black", ls="--", lw=1.0, label="Banda de ruido blanco (95%)")
    ax3.axhline(-conf_band, color="black", ls="--", lw=1.0)
    ax3.axhline(0, color="black", lw=0.7)
    ax3.set_xlabel("Rezago (días)")
    ax3.set_ylabel("Autocorrelación (ACF)")
    ax3.set_title("C. Función de Autocorrelación (ACF): Ausencia de Memoria en Media vs. Larga Memoria en Varianza")
    ax3.legend(loc="upper right", frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_fig1_volatility_clustering.png")
    plt.close(fig)


# ==============================================================================
# BLOQUE 2: JUSTIFICACIÓN DE DEEP LEARNING (RELACIONES NO LINEALES Y REGÍMENES)
# ==============================================================================
def analyze_nonlinearity_and_ols(df_full: pd.DataFrame, X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Cuantifica la brecha entre dependencia lineal (Pearson / MCO) y dependencia
    no lineal (Spearman, Información Mutua, correlación con |feature|, Test RESET).
    """
    log_y = np.log(y + 1e-8)

    mi_raw = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
    mi_log = mutual_info_regression(X, log_y, n_neighbors=5, random_state=42)

    feat_rows = []
    for i, col in enumerate(feature_names):
        x_col = X[:, i]
        r_p, p_p = stats.pearsonr(x_col, y)
        r_abs_p, p_abs_p = stats.pearsonr(np.abs(x_col), y)
        r_s, p_s = stats.spearmanr(x_col, y)

        vif_val = float(variance_inflation_factor(X, i))
        feat_rows.append({
            "Feature": col,
            "Pearson_r": float(r_p),
            "Pearson_p": float(p_p),
            "Pearson_AbsX_r": float(r_abs_p),
            "Pearson_AbsX_p": float(p_abs_p),
            "Spearman_rho": float(r_s),
            "Spearman_p": float(p_s),
            "Mutual_Info_Target": float(mi_raw[i]),
            "Mutual_Info_LogTarget": float(mi_log[i]),
            "VIF": vif_val,
        })

    df_nonlin = pd.DataFrame(feat_rows).sort_values("Mutual_Info_Target", ascending=False).reset_index(drop=True)
    df_nonlin.to_csv(EDA_DIR / "eda_feature_nonlinearity.csv", index=False)

    # Regresión OLS de referencia + Tests de mala especificación funcional
    X_const = sm.add_constant(X)
    ols_fit = sm.OLS(y, X_const).fit(cov_type="HC3")
    ols_plain = sm.OLS(y, X_const).fit()

    reset_res = linear_reset(ols_plain, power=3, use_f=True)
    white_stat, white_p, _, _ = het_white(ols_plain.resid, X_const)

    ols_rows = []
    param_names = ["const"] + feature_names
    for idx, name in enumerate(param_names):
        ols_rows.append({
            "Termino": name,
            "Coef_OLS": float(ols_fit.params[idx]),
            "Std_Err_HC3": float(ols_fit.bse[idx]),
            "t_stat_HC3": float(ols_fit.tvalues[idx]),
            "p_val_HC3": float(ols_fit.pvalues[idx]),
        })

    df_ols = pd.DataFrame(ols_rows)
    df_ols["R2"] = float(ols_plain.rsquared)
    df_ols["R2_Adj"] = float(ols_plain.rsquared_adj)
    df_ols["Ramsey_RESET_F"] = float(reset_res.fvalue)
    df_ols["Ramsey_RESET_p"] = float(reset_res.pvalue)
    df_ols["White_Heterosk_Stat"] = float(white_stat)
    df_ols["White_Heterosk_p"] = float(white_p)
    df_ols.to_csv(EDA_DIR / "eda_ols_diagnostics.csv", index=False)

    # News Impact Curve empírica por deciles de retorno de BTC y SPY
    df_nic = df_full[["BTC_r_oc", "SPY_r_oc", "VIX_log", "target_r2_t1"]].copy()
    df_nic["BTC_Decil"] = pd.qcut(df_nic["BTC_r_oc"], q=10, labels=[f"D{d}" for d in range(1, 11)])
    nic_summary = df_nic.groupby("BTC_Decil", observed=False).agg(
        Observaciones=("target_r2_t1", "count"),
        BTC_r_oc_Medio_pct=("BTC_r_oc", lambda s: float(s.mean() * 100.0)),
        Target_r2_Media_pct2=("target_r2_t1", lambda s: float(s.mean() * 10000.0)),
        Target_r2_Mediana_pct2=("target_r2_t1", lambda s: float(s.median() * 10000.0)),
        Target_r2_Q90_pct2=("target_r2_t1", lambda s: float(s.quantile(0.90) * 10000.0)),
    ).reset_index()
    nic_summary.to_csv(EDA_DIR / "eda_news_impact_deciles.csv", index=False)

    return df_nonlin, df_ols, nic_summary


def plot_nonlinearity_justification(df_full: pd.DataFrame, df_nonlin: pd.DataFrame, nic_summary: pd.DataFrame) -> None:
    """
    Figura 2: Justificación visual de modelos no lineales (News Impact Curve en 'V' asimétrica,
    Información Mutua vs. Correlación Lineal, y régimen VIX x Calendario).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    # Panel A: News Impact Curve Empírica (Deciles de BTC_r_oc -> Varianza en t+1)
    ax1 = axes[0]
    x_pos = np.arange(len(nic_summary))
    ax1.bar(x_pos, nic_summary["Target_r2_Media_pct2"], color="#1f77b4", alpha=0.78, label=r"Media $\mathbb{E}[r_{t+1}^2]$ (%$^2$)")
    ax1.plot(x_pos, nic_summary["Target_r2_Mediana_pct2"] * 4.0, color="#d62728", marker="o", lw=2.0,
             label=r"Mediana $\times 4$ (escala visual)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [f"{d}\n({m:+.1f}%)" for d, m in zip(nic_summary["BTC_Decil"], nic_summary["BTC_r_oc_Medio_pct"])],
        rotation=45, ha="right", fontsize=8
    )
    ax1.set_ylabel(r"Varianza futura $r_{t+1}^2$ (%$^2$)")
    ax1.set_title("A. News Impact Curve Empírica\n(Respuesta en 'V' Asimétrica / Apalancamiento)")
    ax1.legend(loc="upper center", frameon=True, fontsize=8.5)

    # Panel B: Correlación Lineal |Pearson| vs Información Mutua Normalizada
    ax2 = axes[1]
    df_p = df_nonlin.sort_values("Mutual_Info_Target", ascending=True)
    y_pos = np.arange(len(df_p))
    abs_pearson_norm = df_p["Pearson_r"].abs() / max(df_p["Pearson_r"].abs().max(), 1e-8)
    mi_norm = df_p["Mutual_Info_Target"] / max(df_p["Mutual_Info_Target"].max(), 1e-8)

    h = 0.36
    ax2.barh(y_pos - h / 2, abs_pearson_norm, height=h, color="#7f7f7f", label=r"$|r_{\mathrm{Pearson}}|$ (lineal, norm.)")
    ax2.barh(y_pos + h / 2, mi_norm, height=h, color="#2ca02c", label="Información Mutua (no lineal, norm.)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_p["Feature"], fontsize=9)
    ax2.set_xlabel("Intensidad de señal relativa (normalizada [0, 1])")
    ax2.set_title("B. Señal Lineal vs. No Lineal\n(Features con r ≈ 0 pero alta Info. Mutua)")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

    # Panel C: Interacción Régimen VIX x Calendario (Hábiles NYSE vs Fin de Semana/Feriado)
    ax3 = axes[2]
    df_cal = df_full.copy()
    is_weekend_or_holiday = (
        (df_cal["SPY_r_oc"] == 0.0)
        & (df_cal["SPY_r_ho"] == 0.0)
        & (df_cal["SPY_r_lo"] == 0.0)
        & (df_cal["SPY_r_vol"] == 0.0)
    )
    df_cal["Regimen_Calendario"] = np.where(is_weekend_or_holiday, "Fin de semana / Feriado NYSE", "Día hábil NYSE")
    df_cal["Cuartil_VIX"] = pd.qcut(np.exp(df_cal["VIX_log"]), q=4, labels=["Q1 (Calma)", "Q2", "Q3", "Q4 (Estrés)"])

    piv = df_cal.groupby(["Cuartil_VIX", "Regimen_Calendario"], observed=False)["target_r2_t1"].mean().unstack() * 10000.0
    x_q = np.arange(len(piv))
    w_q = 0.36
    ax3.bar(x_q - w_q / 2, piv["Día hábil NYSE"], width=w_q, color="#1f77b4", label="Día hábil NYSE (SPY activo)")
    ax3.bar(x_q + w_q / 2, piv["Fin de semana / Feriado NYSE"], width=w_q, color="#ff7f0e", label="Fin de semana / Feriado (SPY=0)")
    ax3.set_xticks(x_q)
    ax3.set_xticklabels(piv.index, fontsize=9)
    ax3.set_ylabel(r"Varianza media $\mathbb{E}[r_{t+1}^2]$ (%$^2$)")
    ax3.set_title("C. Interacción VIX × Calendario\n(Spillover Macro Asimétrico)")
    ax3.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_fig2_nonlinearity_justification.png")
    plt.close(fig)


# ==============================================================================
# BLOQUE 3: CATÁLOGO DE OUTLIERS CON FECHAS EXACTAS PARA EL INFORME
# ==============================================================================
def build_outliers_catalog(df_full: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identifica los eventos más extremos de volatilidad diaria registrando tanto la
    fecha de los features (t) como la fecha real en que ocurre el shock (t+1),
    junto con el retorno con signo, nivel de VIX y fase experimental.
    """
    df = df_full.copy()
    df["Feature_Date"] = df.index
    # El target_r2_t1 en la fila t corresponde al retorno close-to-close de la fila t+1
    df["Event_Date"] = pd.Series(df.index, index=df.index).shift(-1)
    df["BTC_r_cc_t1"] = df["BTC_r_cc"].shift(-1)

    # Para la última fila del dataset, reconstruimos Event_Date = Feature_Date + 1 día
    df["Event_Date"] = df["Event_Date"].fillna(df["Feature_Date"] + pd.Timedelta(days=1))
    # Signo en t+1 desde BTC_r_oc desplazado si BTC_r_cc_t1 es NaN en la última fila
    fallback_sign = np.sign(df["BTC_r_oc"].shift(-1)).fillna(1.0)
    df["BTC_r_cc_t1"] = df["BTC_r_cc_t1"].fillna(fallback_sign * np.sqrt(df["target_r2_t1"]))

    phase_a_cutoff = pd.Timestamp(PHASE_A_END_DATE)
    df["Phase"] = np.where(df["Feature_Date"] <= phase_a_cutoff, "Fase_A_Seleccion", "Fase_B_Test_Ciego")

    is_non_trading = (
        (df["SPY_r_oc"] == 0.0)
        & (df["SPY_r_ho"] == 0.0)
        & (df["SPY_r_lo"] == 0.0)
        & (df["SPY_r_vol"] == 0.0)
    )
    df["Calendario_t"] = np.where(is_non_trading, "Fin de semana / Feriado", "Hábil NYSE")

    # Umbral de cola extrema (Percentil 99)
    q99 = float(df["target_r2_t1"].quantile(0.99))
    df["Supera_P99"] = df["target_r2_t1"] > q99

    top_df = df.nlargest(top_n, "target_r2_t1").copy()
    top_df["Rank"] = np.arange(1, len(top_df) + 1)

    catalog = pd.DataFrame({
        "Rank": top_df["Rank"].values,
        "Feature_Date_t": pd.to_datetime(top_df["Feature_Date"]).dt.strftime("%Y-%m-%d").values,
        "Event_Date_t1": pd.to_datetime(top_df["Event_Date"]).dt.strftime("%Y-%m-%d").values,
        "Phase": top_df["Phase"].values,
        "Calendario_t": top_df["Calendario_t"].values,
        "Direccion": np.where(top_df["BTC_r_cc_t1"].values < 0, "Crash (-)", "Rally (+)"),
        "BTC_Return_t1_pct": top_df["BTC_r_cc_t1"].values * 100.0,
        "Target_r2_t1": top_df["target_r2_t1"].values,
        "Target_r2_t1_pct2": top_df["target_r2_t1"].values * 10000.0,
        "Ratio_vs_Mediana": top_df["target_r2_t1"].values / float(df["target_r2_t1"].median()),
        "VIX_Level_t": np.exp(top_df["VIX_log"].values),
        "SPY_r_oc_t_pct": top_df["SPY_r_oc"].values * 100.0,
    })

    catalog.to_csv(EDA_DIR / "eda_outliers_catalog.csv", index=False)
    return catalog


def plot_outliers_and_tails(df_full: pd.DataFrame, catalog: pd.DataFrame) -> None:
    """
    Figura 3: Línea temporal de shocks extremos etiquetados con fecha real (t+1)
    y gráfico QQ-Plot demostrando desviación severa respecto a la Normalidad.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Panel A: Serie de varianza realizada t+1 y anotación de Top-8 outliers
    dates = df_full.index
    y_pct2 = df_full["target_r2_t1"] * 10000.0
    q99_pct2 = float(y_pct2.quantile(0.99))

    ax1.plot(dates, y_pct2, color="#1f77b4", lw=0.85, alpha=0.8, label=r"Varianza objetivo $r_{t+1}^2$ (%$^2$)")
    ax1.axhline(q99_pct2, color="#ff7f0e", ls="--", lw=1.2, label=f"Percentil 99 ({q99_pct2:.1f} %²)")

    top_annot = catalog.head(8)
    for idx, row in top_annot.iterrows():
        f_date = pd.to_datetime(row["Feature_Date_t"])
        y_val = row["Target_r2_t1_pct2"]
        ev_str = row["Event_Date_t1"]
        ret_str = f"{row['BTC_Return_t1_pct']:+.1f}%"
        ax1.scatter([f_date], [y_val], color="#d62728", s=35, zorder=5)
        offset_y = 14 if idx % 2 == 0 else -18
        ax1.annotate(
            f"{ev_str}\n({ret_str})",
            xy=(f_date, y_val),
            xytext=(6, offset_y),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#8c1515",
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7, alpha=0.7),
        )

    ax1.set_yscale("log")
    ax1.set_ylabel(r"Varianza realizada $r_{t+1}^2$ (%$^2$, esc. log)")
    ax1.set_title("A. Identificación de Outliers Extremos de Volatilidad (Fecha Real del Evento t+1)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: QQ-Plot de retornos diarios estandarizados vs Normal(0, 1)
    r = df_full["BTC_r_cc"].values
    z_emp = (r - np.mean(r)) / np.std(r, ddof=1)
    (osm, osr), (slope, intercept, _) = stats.probplot(z_emp, dist="norm")

    ax2.scatter(osm, osr, s=8, alpha=0.55, color="#1f77b4", label="Cuantiles empíricos BTC")
    line_x = np.array([osm.min(), osm.max()])
    ax2.plot(line_x, line_x, color="#d62728", lw=1.8, ls="--", label="Identidad Normal N(0, 1)")
    ax2.set_xlabel("Cuantiles teóricos Normal estándar")
    ax2.set_ylabel("Cuantiles empíricos estandarizados")
    ax2.set_title("B. QQ-Plot de Retornos de Bitcoin\n(Colas Pesadas / Leptocurtosis Extrema)")
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_fig3_outliers_and_tails.png")
    plt.close(fig)


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
def run_full_eda(phase: str = "phase_a", top_n_outliers: int = 20):
    df, X, y, _, feature_names = load_processed_data_with_dates(DATA_PATH, phase=phase)
    df_ref = load_returns_reference(DATA_PATH, phase=phase)

    df_full = df.copy()
    df_full["BTC_r_cc"] = df_ref["BTC_r_cc"]

    print("\n" + "=" * 90)
    print(f"ANÁLISIS EXPLORATORIO DE DATOS (EDA) — MODO: {phase.upper()} ({len(df_full)} observaciones)")
    print(f"Rango temporal analizado: {df_full.index.min().date()} -> {df_full.index.max().date()}")
    print("=" * 90)

    # 1. Estacionariedad, colas pesadas y Volatility Clustering
    df_stats = analyze_stationarity_and_tails(df_full, feature_names)
    df_vc = analyze_volatility_clustering(df_full)

    print("\n[1/3] ESTACIONARIEDAD (ADF / KPSS) Y CURTOSIS:")
    disp_stats = ["Variable", "Asimetria", "Exceso_Curtosis", "ADF_p_val", "KPSS_p_val", "Sin_Raiz_Unitaria_ADF", "Regimen_Estable_KPSS"]
    print(df_stats[disp_stats].to_string(index=False))

    print("\n[1/3] TESTS FORMALES DE VOLATILITY CLUSTERING (LJUNG-BOX Y ARCH-LM):")
    print(df_vc.to_string(index=False))

    # 2. Justificación de Deep Learning (No linealidad + OLS + News Impact Curve)
    df_nonlin, df_ols, nic_summary = analyze_nonlinearity_and_ols(df_full, X, y, feature_names)

    print("\n[2/3] DEPENDENCIA LINEAL (PEARSON) VS. NO LINEAL (SPEARMAN / INFORMACIÓN MUTUA) Y VIF:")
    disp_nonlin = ["Feature", "Pearson_r", "Pearson_AbsX_r", "Spearman_rho", "Mutual_Info_Target", "VIF"]
    print(df_nonlin[disp_nonlin].to_string(index=False))

    print("\n[2/3] DIAGNÓSTICO DE ESPECIFICACIÓN LINEAL MCO (OLS):")
    print(f"  • R² = {df_ols['R2'].iloc[0]:.4f} | R² Ajustado = {df_ols['R2_Adj'].iloc[0]:.4f}")
    print(f"  • Test RESET de Ramsey (no linealidad): F = {df_ols['Ramsey_RESET_F'].iloc[0]:.2f}, "
          f"p-val = {df_ols['Ramsey_RESET_p'].iloc[0]:.3e}")
    print(f"  • Test de Heterocedasticidad de White:  Stat = {df_ols['White_Heterosk_Stat'].iloc[0]:.2f}, "
          f"p-val = {df_ols['White_Heterosk_p'].iloc[0]:.3e}")

    print("\n[2/3] NEWS IMPACT CURVE EMPÍRICA POR DECILES DE RETORNO BTC:")
    print(nic_summary.to_string(index=False))

    # 3. Catálogo de Outliers con fechas reales (t y t+1)
    catalog = build_outliers_catalog(df_full, top_n=top_n_outliers)
    print(f"\n[3/3] TOP-{top_n_outliers} OUTLIERS EXTREMOS DE VOLATILIDAD (CON FECHA REAL DEL EVENTO t+1):")
    print(catalog.to_string(index=False))

    # 4. Generación de figuras
    plot_volatility_clustering(df_full, catalog)
    plot_nonlinearity_justification(df_full, df_nonlin, nic_summary)
    plot_outliers_and_tails(df_full, catalog)

    print("\n" + "-" * 90)
    print(f"Tablas CSV exportadas en: {EDA_DIR.resolve()}")
    print(f"Figuras PNG exportadas en: {FIG_DIR.resolve()}")
    print("-" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Econométrico y de Deep Learning para Volatilidad BTC")
    parser.add_argument(
        "--phase",
        type=str,
        default="phase_a",
        choices=["phase_a", "all"],
        help="Usar 'phase_a' (<= 2025-12-25) para blindar Fase B, o 'all' para el período completo.",
    )
    parser.add_argument("--top-outliers", type=int, default=20, help="Cantidad de eventos extremos a catalogar.")
    args = parser.parse_args()
    run_full_eda(phase=args.phase, top_n_outliers=args.top_outliers)