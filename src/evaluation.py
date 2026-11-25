import numpy as np
import pandas as pd
from scipy import stats


def compute_qlike_vector(y_true, sigma2_pred, eps: float = 1e-6) -> np.ndarray:
    """
    Calcula la pérdida QLIKE diaria vectorizada de forma numéricamente estable:
        QLIKE(y, sigma^2) = y / sigma^2 - ln(y / sigma^2) - 1
    """
    y = np.clip(np.asarray(y_true, dtype=np.float64), eps, None)
    s2 = np.clip(np.asarray(sigma2_pred, dtype=np.float64), eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0


def get_significance_code(p_val: float) -> str:
    """Convención estándar de significancia econométrica."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.10:
        return "·"
    return "n.s."


def diebold_mariano_full(loss_baseline, loss_candidate, h: int = 1, hac_lags=None) -> dict:
    """
    Test de Diebold-Mariano (1995) con estimador HAC de Newey-West (núcleo de Bartlett)
    y ajuste de muestra finita de Harvey, Leybourne y Newbold (1997).

    Parámetros:
        hac_lags:
            - None (default): truncamiento canónico HLN en max_lag = h - 1.
            - "newey_west": ancho de banda automático L = floor(T^(1/3)).
            - int >= 0: número explícito de rezagos de Bartlett.

    Convención de signo:
        d_t = loss_baseline_t - loss_candidate_t
        d_mean > 0 (y DM_stat > 0) indica que el candidato tiene MENOR pérdida que el baseline.
    """
    d = np.asarray(loss_baseline, dtype=np.float64) - np.asarray(loss_candidate, dtype=np.float64)
    T = len(d)
    if T == 0:
        raise ValueError("Los vectores de pérdida no pueden estar vacíos.")

    if hac_lags is None:
        max_lag = max(int(h) - 1, 0)
    elif isinstance(hac_lags, str) and hac_lags.lower().strip() == "newey_west":
        max_lag = int(np.floor(T ** (1.0 / 3.0)))
    else:
        max_lag = max(int(hac_lags), 0)
    max_lag = min(max_lag, max(T - 1, 0))

    mean_d = float(np.mean(d))
    d_centered = d - mean_d

    gamma0 = float(np.dot(d_centered, d_centered) / T)
    acf1_d = (
        float(np.dot(d_centered[1:], d_centered[:-1]) / (T * gamma0))
        if (T > 1 and gamma0 > 1e-15)
        else 0.0
    )

    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        weight = 1.0 - (k / (max_lag + 1.0))
        cov_k = float(np.dot(d_centered[k:], d_centered[:-k]) / T)
        gamma_sum += 2.0 * weight * cov_k

    var_d = gamma0 + gamma_sum
    if var_d <= 1e-12:
        var_d = float(np.var(d, ddof=1))

    se_d = float(np.sqrt(var_d / T)) if var_d > 0.0 else 0.0
    if se_d <= 1e-15:
        return {
            "T": T,
            "diff_mean": mean_d,
            "se_hac": 0.0,
            "dm_stat": 0.0,
            "p_val_bilateral": 1.0,
            "p_val_unilateral": 0.5,
            "hac_lags_used": max_lag,
            "acf1_d": acf1_d,
        }

    dm_stat = mean_d / se_d

    # Factor de corrección de Harvey, Leybourne y Newbold (1997)
    hln_adj = np.sqrt((T + 1.0 - 2.0 * h + (h * (h - 1.0)) / T) / T)
    dm_hln = float(dm_stat * hln_adj)

    p_val_two = float(2.0 * stats.t.sf(np.abs(dm_hln), df=T - 1))
    p_val_one = float(stats.t.sf(dm_hln, df=T - 1))

    return {
        "T": T,
        "diff_mean": mean_d,
        "se_hac": se_d,
        "dm_stat": dm_hln,
        "p_val_bilateral": p_val_two,
        "p_val_unilateral": p_val_one,
        "hac_lags_used": max_lag,
        "acf1_d": acf1_d,
    }


def diebold_mariano_test(loss_baseline, loss_candidate, h: int = 1, hac_lags=None):
    """Wrapper compatible que retorna (dm_stat_HLN, p_val_bilateral)."""
    res = diebold_mariano_full(loss_baseline, loss_candidate, h=h, hac_lags=hac_lags)
    return res["dm_stat"], res["p_val_bilateral"]


def validate_predictions_integrity(
    df_preds: pd.DataFrame,
    df_proc: pd.DataFrame,
    group_cols=None,
    expected_dates: pd.DatetimeIndex = None,
) -> None:
    """
    Contrato de integridad temporal antes de exportar o graficar predicciones:
    1. Sin NaNs ni valores no positivos en sigma2_pred / y_true.
    2. Sin fechas duplicadas dentro de cada grupo (modelo / config_id / seed) y orden creciente.
    3. Coincidencia exacta entre el 'y_true' registrado y 'target_r2_t1' en df_proc[Date].
    4. Si se provee expected_dates, verifica igualdad exacta del conjunto de fechas.
    """
    if df_preds.empty:
        raise ValueError("El DataFrame de predicciones está vacío.")

    required_cols = {"Date", "y_true"}
    if not required_cols.issubset(df_preds.columns):
        raise ValueError(f"Faltan columnas requeridas en predicciones: {required_cols - set(df_preds.columns)}")

    pred_col = "sigma2_pred" if "sigma2_pred" in df_preds.columns else "sigma2_pred_mean"
    if pred_col not in df_preds.columns:
        raise ValueError("No se encontró columna de varianza predicha ('sigma2_pred' o 'sigma2_pred_mean').")

    if df_preds[["Date", "y_true", pred_col]].isna().any().any():
        raise ValueError("Existen valores NaN en Date, y_true o sigma2_pred.")

    if not np.isfinite(df_preds[["y_true", pred_col]].values).all():
        raise ValueError("Existen valores infinitos en y_true o sigma2_pred.")

    if (df_preds[pred_col] <= 0.0).any():
        raise ValueError("Existen predicciones de varianza <= 0.0.")

    dates_series = pd.to_datetime(df_preds["Date"]).dt.tz_localize(None).dt.normalize()
    if not dates_series.isin(df_proc.index).all():
        missing = dates_series[~dates_series.isin(df_proc.index)].unique()
        raise ValueError(f"Hay fechas en predicciones que no existen en processed_data: {missing[:5]}")

    expected_y = df_proc.loc[dates_series, "target_r2_t1"].values
    if not np.allclose(df_preds["y_true"].values, expected_y, rtol=1e-6, atol=1e-10):
        max_diff = float(np.max(np.abs(df_preds["y_true"].values - expected_y)))
        raise ValueError(f"Desalineación en y_true vs processed_data['target_r2_t1'] (max_diff={max_diff:.3e}).")

    if group_cols is None:
        group_cols = [c for c in ["model", "architecture", "config_id", "seed"] if c in df_preds.columns]

    if group_cols:
        for keys, grp in df_preds.groupby(group_cols, sort=False):
            g_dates = pd.to_datetime(grp["Date"])
            if g_dates.duplicated().any():
                raise ValueError(f"Fechas duplicadas detectadas en grupo {keys}.")
            if not g_dates.is_monotonic_increasing:
                raise ValueError(f"Las fechas no están ordenadas cronológicamente en grupo {keys}.")
            if expected_dates is not None:
                exp_idx = pd.DatetimeIndex(expected_dates).sort_values()
                if not pd.DatetimeIndex(g_dates).equals(exp_idx):
                    raise ValueError(
                        f"El grupo {keys} tiene {len(g_dates)} fechas, pero se esperaban {len(exp_idx)}."
                    )


def analyze_covid_regime(
    df_preds: pd.DataFrame,
    start_date: str = "2020-02-01",
    end_date: str = "2020-06-30",
) -> dict:
    """
    Filtra y analiza el desempeño del modelo en la ventana del shock COVID-19 (Feb-Jun 2020).
    """
    df = df_preds.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    df_covid = df.loc[mask].copy()

    pred_col = "sigma2_pred" if "sigma2_pred" in df_covid.columns else "sigma2_pred_mean"
    qlike_col = "qlike" if "qlike" in df_covid.columns else "qlike_ensemble"

    return {
        "num_samples": len(df_covid),
        "mean_qlike": float(df_covid[qlike_col].mean()) if len(df_covid) > 0 else np.nan,
        "median_qlike": float(df_covid[qlike_col].median()) if len(df_covid) > 0 else np.nan,
        "std_qlike": float(df_covid[qlike_col].std(ddof=1)) if len(df_covid) > 1 else np.nan,
        "max_y_true": float(df_covid["y_true"].max()) if len(df_covid) > 0 else np.nan,
        "max_sigma2_pred": float(df_covid[pred_col].max()) if len(df_covid) > 0 else np.nan,
        "subset_df": df_covid,
    }