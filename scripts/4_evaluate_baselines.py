import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

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
from src.econometrics import DEFAULT_EXOG_COLS, EGARCHXModel
from src.evaluation import compute_qlike_vector, validate_predictions_integrity

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_best_egarch_x_config() -> dict:
    """
    Carga la especificación óptima de EGARCH-X congelada en Fase A por 3_tune_egarch_x.py.
    """
    spec_path = RESULTS_DIR / "egarch_x_best_spec.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Especificación EGARCH-X cargada desde {spec_path.name}: {cfg['spec']}")
        return cfg

    print("[Aviso] No se encontró egarch_x_best_spec.json. Usando fallback EGARCH(1,3)-X_GED.")
    return {
        "spec": "EGARCH(1,3)-X_GED",
        "p": 1,
        "o": 1,
        "q": 3,
        "dist": "ged",
        "exog_cols": list(DEFAULT_EXOG_COLS),
    }


def evaluate_all_baselines(
    train_window_days: int = 1095,
    val_window_days: int = 182,
    step_days: int = 182,
):
    """
    Ejecuta el Walk-Forward completo (Fase A: Folds 0–13 + Fase B Test Ciego: Folds 14–15)
    para los 3 modelos de referencia:
      1. Benchmark_Promedio: Varianza incondicional móvil de los últimos 1,095 días.
      2. EGARCH(1,3)_GED: Modelo econométrico univariado asimétrico con colas pesadas.
      3. EGARCH-X óptimo: Modelo multivariado con covariables exógenas en log-varianza.
    """
    df_proc, _, y, _, _ = load_processed_data_with_dates(DATA_PATH, phase="all")
    df_ref = load_returns_reference(DATA_PATH, phase="all")
    total_obs = len(y)

    # Retorno exacto close-to-close en t en escala porcentual (100 * r_t)
    returns_pct = (df_ref["BTC_r_cc"] * 100.0).rename("BTC_Returns_Pct")

    # Configuración congelada del EGARCH-X ganador en Fase A
    egarch_x_cfg = _load_best_egarch_x_config()
    exog_cols = egarch_x_cfg["exog_cols"]
    egarch_x_name = egarch_x_cfg["spec"]
    df_exog = df_proc[exog_cols].copy()

    phase_a_cutoff = pd.Timestamp(PHASE_A_END_DATE)
    all_predictions = []
    all_folds_detail = []

    print("\n" + "=" * 92)
    print("EVALUACIÓN UNIFICADA DE BASELINES (FASE A: 2019–2025 | FASE B CIEGA: 2026)")
    print("=" * 92)
    print(f"Observaciones totales: {total_obs} | Ventana entrenamiento: {train_window_days}d | Bloque test: {val_window_days}d")
    print(f"Modelo multivariado seleccionado: {egarch_x_name} | Exógenas: {exog_cols}\n")

    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_obs:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_obs)

        if (val_end - val_start) < 90:
            break

        val_dates = df_proc.index[val_start:val_end]
        n_samples = val_end - val_start
        phase_label = "Fase_A_Seleccion" if val_dates[-1] <= phase_a_cutoff else "Fase_B_Test_Ciego"

        # Datos en escala decimal
        y_train_dec = y[train_start:train_end]
        y_val_dec = y[val_start:val_end]

        # Ventana causal [train_start : val_end] para filtrado recursivo sin lookahead
        r_hist_oos = returns_pct.iloc[train_start:val_end]
        X_hist_oos = df_exog.iloc[train_start:val_end].values
        oos_offset = train_end - train_start
        first_oos_date = r_hist_oos.index[oos_offset]

        print(
            f">> Fold {fold_idx:02d} [{phase_label}] "
            f"({val_dates[0].strftime('%Y-%m-%d')} -> {val_dates[-1].strftime('%Y-%m-%d')}) | {n_samples} días"
        )

        # ---------------------------------------------------------------------
        # 1. BASELINE TRIVIAL: PROMEDIO MÓVIL (VARIANZA INCONDICIONAL)
        # ---------------------------------------------------------------------
        rolling_var_dec = float(np.mean(y_train_dec))
        pred_promedio_dec = np.full(n_samples, rolling_var_dec, dtype=np.float64)
        qlike_promedio = compute_qlike_vector(y_val_dec, pred_promedio_dec)

        all_folds_detail.append({
            "phase": phase_label,
            "model": "Benchmark_Promedio",
            "fold": fold_idx,
            "val_start": val_dates[0].strftime("%Y-%m-%d"),
            "val_end": val_dates[-1].strftime("%Y-%m-%d"),
            "samples": n_samples,
            "mean_qlike": float(np.mean(qlike_promedio)),
            "median_qlike": float(np.median(qlike_promedio)),
            "std_daily": float(np.std(qlike_promedio, ddof=1)),
            "converged": True,
        })

        for d, yt, sp, ql in zip(val_dates, y_val_dec, pred_promedio_dec, qlike_promedio):
            all_predictions.append({
                "Date": pd.to_datetime(d),
                "phase": phase_label,
                "fold": fold_idx,
                "model": "Benchmark_Promedio",
                "y_true": float(yt),
                "sigma2_pred": float(sp),
                "sigma2_pred_pct": float(sp * 10000.0),
                "qlike": float(ql),
            })

        # ---------------------------------------------------------------------
        # 2. BASELINE ECONOMÉTRICO 1: EGARCH(1,3)_GED (Univariado)
        # ---------------------------------------------------------------------
        am_ged = arch_model(
            r_hist_oos,
            mean="Constant",
            vol="EGARCH",
            p=1,
            o=1,
            q=3,
            dist="ged",
        )
        try:
            # En arch, last_obs funciona como un slice (excluye first_oos_date),
            # ajustando MLE estrictamente sobre los 1,095 días previos.
            res_ged = am_ged.fit(last_obs=first_oos_date, disp="off", show_warning=False)
            f_ged = res_ged.forecast(horizon=1, start=first_oos_date, align="origin")
            s2_ged_pct = f_ged.variance["h.1"].loc[val_dates].values
            s2_ged_dec = s2_ged_pct / 10000.0
            if not np.isfinite(s2_ged_dec).all() or (s2_ged_dec <= 0).any():
                raise ValueError("Pronóstico no finito o no positivo en EGARCH(1,3)_GED.")
            qlike_ged = compute_qlike_vector(y_val_dec, s2_ged_dec)
            conv_ged = bool(res_ged.convergence_flag == 0)
        except Exception as e:
            print(f"   [!] Fallback en EGARCH(1,3)_GED Fold {fold_idx}: {e}")
            s2_ged_dec = pred_promedio_dec.copy()
            s2_ged_pct = pred_promedio_dec * 10000.0
            qlike_ged = qlike_promedio.copy()
            conv_ged = False

        all_folds_detail.append({
            "phase": phase_label,
            "model": "EGARCH(1,3)_GED",
            "fold": fold_idx,
            "val_start": val_dates[0].strftime("%Y-%m-%d"),
            "val_end": val_dates[-1].strftime("%Y-%m-%d"),
            "samples": n_samples,
            "mean_qlike": float(np.mean(qlike_ged)),
            "median_qlike": float(np.median(qlike_ged)),
            "std_daily": float(np.std(qlike_ged, ddof=1)),
            "converged": conv_ged,
        })

        for d, yt, sp, spp, ql in zip(val_dates, y_val_dec, s2_ged_dec, s2_ged_pct, qlike_ged):
            all_predictions.append({
                "Date": pd.to_datetime(d),
                "phase": phase_label,
                "fold": fold_idx,
                "model": "EGARCH(1,3)_GED",
                "y_true": float(yt),
                "sigma2_pred": float(sp),
                "sigma2_pred_pct": float(spp),
                "qlike": float(ql),
            })

        # ---------------------------------------------------------------------
        # 3. BASELINE ECONOMÉTRICO 2: EGARCH-X ÓPTIMO (Multivariado en Varianza)
        # ---------------------------------------------------------------------
        model_x = EGARCHXModel(
            p=int(egarch_x_cfg["p"]),
            o=int(egarch_x_cfg.get("o", 1)),
            q=int(egarch_x_cfg["q"]),
            dist=str(egarch_x_cfg["dist"]),
        )
        try:
            r_train_arr = returns_pct.iloc[train_start:train_end].values
            X_train_arr = df_exog.iloc[train_start:train_end].values
            model_x.fit(r_train_arr, X_train_arr)
            if model_x.params_ is None:
                raise RuntimeError("MLE EGARCH-X sin parámetros válidos.")

            s2_x_pct = model_x.forecast_oos_1step(r_hist_oos.values, X_hist_oos, oos_start_idx=oos_offset)
            s2_x_dec = s2_x_pct / 10000.0
            if not np.isfinite(s2_x_dec).all() or (s2_x_dec <= 0).any():
                raise ValueError("Pronóstico no finito o no positivo en EGARCH-X.")

            qlike_x = compute_qlike_vector(y_val_dec, s2_x_dec)
            conv_x = model_x.converged_
        except Exception as e:
            print(f"   [!] Fallback en {egarch_x_name} Fold {fold_idx}: {e}")
            s2_x_dec = pred_promedio_dec.copy()
            s2_x_pct = pred_promedio_dec * 10000.0
            qlike_x = qlike_promedio.copy()
            conv_x = False

        all_folds_detail.append({
            "phase": phase_label,
            "model": egarch_x_name,
            "fold": fold_idx,
            "val_start": val_dates[0].strftime("%Y-%m-%d"),
            "val_end": val_dates[-1].strftime("%Y-%m-%d"),
            "samples": n_samples,
            "mean_qlike": float(np.mean(qlike_x)),
            "median_qlike": float(np.median(qlike_x)),
            "std_daily": float(np.std(qlike_x, ddof=1)),
            "converged": conv_x,
        })

        for d, yt, sp, spp, ql in zip(val_dates, y_val_dec, s2_x_dec, s2_x_pct, qlike_x):
            all_predictions.append({
                "Date": pd.to_datetime(d),
                "phase": phase_label,
                "fold": fold_idx,
                "model": egarch_x_name,
                "y_true": float(yt),
                "sigma2_pred": float(sp),
                "sigma2_pred_pct": float(spp),
                "qlike": float(ql),
            })

        print(
            f"   • Promedio: {np.mean(qlike_promedio):.4f} (Med: {np.median(qlike_promedio):.4f}) | "
            f"EGARCH_GED: {np.mean(qlike_ged):.4f} (Med: {np.median(qlike_ged):.4f}) | "
            f"{egarch_x_name}: {np.mean(qlike_x):.4f} (Med: {np.median(qlike_x):.4f})"
        )

        start_idx += step_days
        fold_idx += 1

    # Consolidación final y validación estricta de integridad temporal
    df_preds_all = pd.DataFrame(all_predictions)
    df_folds_all = pd.DataFrame(all_folds_detail)

    validate_predictions_integrity(df_preds_all, df_proc, group_cols=["model"])

    # Resumen por fase (Fase A, Fase B Ciega y Muestra Completa)
    model_names = ["Benchmark_Promedio", "EGARCH(1,3)_GED", egarch_x_name]
    phase_scopes = [
        ("Fase_A_Seleccion", df_folds_all[df_folds_all["phase"] == "Fase_A_Seleccion"], df_preds_all[df_preds_all["phase"] == "Fase_A_Seleccion"]),
        ("Fase_B_Test_Ciego", df_folds_all[df_folds_all["phase"] == "Fase_B_Test_Ciego"], df_preds_all[df_preds_all["phase"] == "Fase_B_Test_Ciego"]),
        ("Muestra_Completa", df_folds_all, df_preds_all),
    ]

    summary_rows = []
    for scope_name, scope_folds, scope_preds in phase_scopes:
        if scope_preds.empty:
            continue
        for m_name in model_names:
            sub_f = scope_folds[scope_folds["model"] == m_name]
            sub_p = scope_preds[scope_preds["model"] == m_name]
            summary_rows.append({
                "phase": scope_name,
                "model": m_name,
                "mean_qlike_macro": float(sub_f["mean_qlike"].mean()),
                "std_folds": float(sub_f["mean_qlike"].std(ddof=1)) if len(sub_f) > 1 else 0.0,
                "median_qlike_macro": float(sub_f["median_qlike"].mean()),
                "mean_qlike_micro": float(sub_p["qlike"].mean()),
                "median_qlike_micro": float(sub_p["qlike"].median()),
                "total_observations": int(len(sub_p)),
                "converged_folds": int(sub_f["converged"].sum()),
                "total_folds": int(len(sub_f)),
            })

    df_summary = pd.DataFrame(summary_rows)

    # Persistencia de artefactos
    df_preds_all.to_parquet(RESULTS_DIR / "baselines_predictions.parquet", index=False)
    df_folds_all.to_csv(RESULTS_DIR / "baselines_folds_detail.csv", index=False)
    df_summary.to_csv(RESULTS_DIR / "baselines_summary.csv", index=False)

    print("\n" + "=" * 92)
    print("RESUMEN DE BASELINES POR FASE (GUARDADO EN results/baselines_summary.csv):")
    print("=" * 92)
    print(df_summary.to_string(index=False))
    print(f"\nPredicciones exportadas a: {RESULTS_DIR / 'baselines_predictions.parquet'}")


if __name__ == "__main__":
    evaluate_all_baselines()