import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_processed_data_with_dates, load_returns_reference
from src.econometrics import DEFAULT_EXOG_COLS, EGARCHXModel
from src.evaluation import compute_qlike_vector, validate_predictions_integrity

DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_egarch_x_walk_forward_phase_a(
    df_proc: pd.DataFrame,
    returns_pct: pd.Series,
    df_exog: pd.DataFrame,
    p: int,
    q: int,
    dist: str,
    train_window_days: int = 1095,
    val_window_days: int = 182,
    step_days: int = 182,
):
    """
    Ejecuta el Walk-Forward de EGARCH(p, 1, q)-X estrictamente sobre Fase A (2016-2025)
    sin lookahead bias y guarda las predicciones diarias y parámetros por fold.
    """
    total_obs = len(returns_pct)
    y_dec_all = df_proc["target_r2_t1"].values
    dates_all = df_proc.index
    exog_cols = list(df_exog.columns)

    pred_rows = []
    fold_rows = []
    params_rows = []

    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_obs:
        train_start = start_idx
        train_end = start_idx + train_window_days
        test_start = train_end
        test_end = min(test_start + val_window_days, total_obs)

        if (test_end - test_start) < 90:
            break

        # 1. Ajuste MLE estrictamente con observaciones in-sample [train_start : train_end]
        r_train = returns_pct.iloc[train_start:train_end].values
        X_train = df_exog.iloc[train_start:train_end].values

        model = EGARCHXModel(p=p, o=1, q=q, dist=dist)
        try:
            model.fit(r_train, X_train)
            conv_ok = model.converged_
        except Exception:
            conv_ok = False

        # 2. Pronóstico causal 1-paso adelante sobre [test_start : test_end]
        r_hist_oos = returns_pct.iloc[train_start:test_end].values
        X_hist_oos = df_exog.iloc[train_start:test_end].values
        oos_offset = train_end - train_start

        y_val_dec = y_dec_all[test_start:test_end]
        val_dates = dates_all[test_start:test_end]

        try:
            if model.params_ is None:
                raise RuntimeError("MLE sin parámetros válidos.")
            s2_pred_pct = model.forecast_oos_1step(r_hist_oos, X_hist_oos, oos_start_idx=oos_offset)
            s2_pred_dec = s2_pred_pct / 10000.0
        except Exception:
            # Fallback conservador a varianza incondicional in-sample si fallara MLE
            fallback_var = float(np.mean(y_dec_all[train_start:train_end]))
            s2_pred_dec = np.full(len(y_val_dec), fallback_var, dtype=np.float64)
            s2_pred_pct = s2_pred_dec * 10000.0
            conv_ok = False

        qlike_vec = compute_qlike_vector(y_val_dec, s2_pred_dec)

        fold_rows.append({
            "fold": fold_idx,
            "val_start": val_dates[0].strftime("%Y-%m-%d"),
            "val_end": val_dates[-1].strftime("%Y-%m-%d"),
            "samples": len(qlike_vec),
            "mean_qlike": float(np.mean(qlike_vec)),
            "median_qlike": float(np.median(qlike_vec)),
            "std_daily": float(np.std(qlike_vec, ddof=1)),
            "aic": float(model.aic_),
            "bic": float(model.bic_),
            "converged": conv_ok,
        })

        if model.params_ is not None:
            p_dict = model.get_params_dict(exog_cols)
            p_dict.update({
                "fold": fold_idx,
                "val_start": val_dates[0].strftime("%Y-%m-%d"),
                "val_end": val_dates[-1].strftime("%Y-%m-%d"),
                "converged": conv_ok,
                "mean_qlike": float(np.mean(qlike_vec)),
            })
            params_rows.append(p_dict)

        for d, yt, spd, spp, ql in zip(val_dates, y_val_dec, s2_pred_dec, s2_pred_pct, qlike_vec):
            pred_rows.append({
                "Date": pd.to_datetime(d),
                "phase": "Fase_A_Seleccion",
                "fold": fold_idx,
                "spec": f"EGARCH({p},{q})-X_{dist.upper()}",
                "y_true": float(yt),
                "sigma2_pred": float(spd),
                "sigma2_pred_pct": float(spp),
                "qlike": float(ql),
            })

        start_idx += step_days
        fold_idx += 1

    df_folds = pd.DataFrame(fold_rows)
    df_preds = pd.DataFrame(pred_rows)
    df_params = pd.DataFrame(params_rows)

    failed_folds = int((~df_folds["converged"]).sum())
    total_folds = int(len(df_folds))

    return {
        "mean_qlike_macro": float(df_folds["mean_qlike"].mean()),
        "mean_qlike_micro": float(df_preds["qlike"].mean()),
        "median_qlike_micro": float(df_preds["qlike"].median()),
        "std_folds": float(df_folds["mean_qlike"].std(ddof=1)),
        "std_daily": float(df_preds["qlike"].std(ddof=1)),
        "aic_mean": float(df_folds["aic"].mean()),
        "bic_mean": float(df_folds["bic"].mean()),
        "failed_folds": failed_folds,
        "total_folds": total_folds,
        "df_preds": df_preds,
        "df_folds": df_folds,
        "df_params": df_params,
    }


def find_optimal_egarch_x_by_qlike(
    train_window_days: int = 1095,
    val_window_days: int = 182,
    exog_cols: list = None,
):
    """
    Grid Search de EGARCH(p, 1, q)-X evaluado exclusivamente sobre Fase A (2019-2025)
    para preservar el test ciego de 2026 (Fase B).
    """
    if exog_cols is None:
        exog_cols = list(DEFAULT_EXOG_COLS)

    # Carga estricta hasta PHASE_A_END_DATE (2025-12-25)
    df_proc, _, _, _, _ = load_processed_data_with_dates(DATA_PATH, phase="phase_a")
    df_ref = load_returns_reference(DATA_PATH, phase="phase_a")

    # Retorno exacto close-to-close en t en escala porcentual (100 * r_t)
    returns_pct = (df_ref["BTC_r_cc"] * 100.0).rename("BTC_Returns_Pct")
    df_exog = df_proc[exog_cols].copy()

    p_orders = [1, 2]
    q_orders = [1, 2, 3]
    distributions = ["ged", "t", "normal"]

    grid_results = []
    all_grid_preds = []
    params_by_spec = {}

    print("\n" + "=" * 92)
    print("GRID SEARCH DE EGARCH(p,1,q)-X EN VARIANZA — FASE A SELECCIÓN (2016–2025)")
    print("=" * 92)
    print(f"Observaciones Fase A: {len(df_proc)} | Ventana inicial: {train_window_days}d | Bloque test: {val_window_days}d")
    print(f"Covariables exógenas en ecuación de log-varianza: {exog_cols}\n")

    for p, q, dist in itertools.product(p_orders, q_orders, distributions):
        spec_name = f"EGARCH({p},{q})-X_{dist.upper()}"
        t0 = time.time()
        print(f"Evaluando {spec_name:24s}...", end=" ", flush=True)

        res = run_egarch_x_walk_forward_phase_a(
            df_proc=df_proc,
            returns_pct=returns_pct,
            df_exog=df_exog,
            p=p,
            q=q,
            dist=dist,
            train_window_days=train_window_days,
            val_window_days=val_window_days,
            step_days=val_window_days,
        )
        elapsed = time.time() - t0

        print(
            f"Mean QLIKE: {res['mean_qlike_micro']:.4f} | "
            f"Mediana: {res['median_qlike_micro']:.4f} | "
            f"BIC: {res['bic_mean']:.1f} | "
            f"Fails: {res['failed_folds']}/{res['total_folds']} ({elapsed:.1f}s)"
        )

        grid_results.append({
            "spec": spec_name,
            "p": p,
            "q": q,
            "dist": dist,
            "mean_qlike": res["mean_qlike_micro"],
            "mean_qlike_macro": res["mean_qlike_macro"],
            "median_qlike": res["median_qlike_micro"],
            "std_folds": res["std_folds"],
            "std_daily": res["std_daily"],
            "aic_mean": res["aic_mean"],
            "bic_mean": res["bic_mean"],
            "failed_folds": res["failed_folds"],
            "total_folds": res["total_folds"],
        })
        all_grid_preds.append(res["df_preds"])
        params_by_spec[spec_name] = res["df_params"]

    results_df = pd.DataFrame(grid_results).sort_values(by="mean_qlike").reset_index(drop=True)

    print("\n" + "=" * 92)
    print("RANKING FINAL EGARCH-X SEGÚN QLIKE OUT-OF-SAMPLE (FASE A: 2019–2025):")
    print("=" * 92)
    disp_cols = ["spec", "mean_qlike", "median_qlike", "std_folds", "bic_mean", "failed_folds"]
    print(results_df[disp_cols].to_string(index=False))

    best_model = results_df.iloc[0]
    best_spec = best_model["spec"]

    print("\n" + "-" * 92)
    print(f"ESPECIFICACIÓN ÓPTIMA CONGELADA PARA FASE B: {best_spec}")
    print(f"Mean QLIKE: {best_model['mean_qlike']:.4f} | Median QLIKE: {best_model['median_qlike']:.4f}")
    print("-" * 92)

    # Validación estricta de integridad temporal antes de guardar predicciones diarias
    master_grid_preds = pd.concat(all_grid_preds, ignore_index=True)
    validate_predictions_integrity(master_grid_preds, df_proc, group_cols=["spec"])

    # Persistencia completa
    results_df.to_csv(RESULTS_DIR / "egarch_x_qlike_grid_search.csv", index=False)
    master_grid_preds.to_parquet(RESULTS_DIR / "egarch_x_grid_predictions.parquet", index=False)

    best_params_df = params_by_spec[best_spec]
    best_params_df.to_csv(RESULTS_DIR / "egarch_x_best_params_by_fold.csv", index=False)

    best_spec_payload = {
        "spec": best_spec,
        "p": int(best_model["p"]),
        "o": 1,
        "q": int(best_model["q"]),
        "dist": str(best_model["dist"]),
        "exog_cols": exog_cols,
        "phase_a_mean_qlike": float(best_model["mean_qlike"]),
        "phase_a_median_qlike": float(best_model["median_qlike"]),
    }
    with open(RESULTS_DIR / "egarch_x_best_spec.json", "w", encoding="utf-8") as f:
        json.dump(best_spec_payload, f, indent=2)

    print("\nArtefactos persistidos:")
    print(f"  - {RESULTS_DIR / 'egarch_x_qlike_grid_search.csv'}")
    print(f"  - {RESULTS_DIR / 'egarch_x_grid_predictions.parquet'}")
    print(f"  - {RESULTS_DIR / 'egarch_x_best_params_by_fold.csv'}")
    print(f"  - {RESULTS_DIR / 'egarch_x_best_spec.json'}")

    return best_model


if __name__ == "__main__":
    find_optimal_egarch_x_by_qlike()