import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    EXPECTED_FEATURE_COLS,
    VolatilidadCNN,
    VolatilidadGRU,
    VolatilidadLSTM,
    compute_qlike_vector,
    load_processed_data_with_dates,
    run_walk_forward,
    validate_predictions_integrity,
)

DEFAULT_SEEDS = [42, 123, 456, 789, 7122001]
DATA_PATH = ROOT / "data" / "processed" / "processed_data.csv"
RESULTS_DIR = ROOT / "results"
SWEEPS_DIR = RESULTS_DIR / "sweeps"
PARTIAL_DIR = RESULTS_DIR / "partial"
RUNS_DIR = PARTIAL_DIR / "runs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ARCH_REGISTRY = {
    "CNN_1D": {
        "model_class": VolatilidadCNN,
        "history_file": SWEEPS_DIR / "sweep_cnn_history.csv",
        "int_params": ["timesteps", "hidden_size", "kernel_size", "num_layers", "batch_size"],
        "float_params": ["dropout_rate", "learning_rate", "weight_decay"],
    },
    "GRU": {
        "model_class": VolatilidadGRU,
        "history_file": SWEEPS_DIR / "sweep_gru_history.csv",
        "int_params": ["timesteps", "hidden_size", "num_layers", "batch_size"],
        "float_params": ["dropout_rate", "learning_rate", "weight_decay"],
    },
    "LSTM": {
        "model_class": VolatilidadLSTM,
        "history_file": SWEEPS_DIR / "sweep_lstm_history.csv",
        "int_params": ["timesteps", "hidden_size", "num_layers", "batch_size"],
        "float_params": ["dropout_rate", "learning_rate", "weight_decay"],
    },
}


def _count_trainable_params(model_class, cfg: dict, input_size: int = len(EXPECTED_FEATURE_COLS)) -> int:
    """Instancia el modelo estrictamente en CPU (sin inicializar CUDA) para contar parámetros."""
    kwargs = {
        "input_size": input_size,
        "hidden_size": cfg["hidden_size"],
        "dropout_rate": cfg["dropout_rate"],
        "num_layers": cfg["num_layers"],
    }
    if model_class.__name__ == "VolatilidadCNN":
        kwargs["kernel_size"] = cfg["kernel_size"]
    model = model_class(**kwargs)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _get_phase_a_sanity_threshold() -> float:
    base_summary_path = RESULTS_DIR / "baselines_summary.csv"
    if base_summary_path.exists():
        df_b = pd.read_csv(base_summary_path)
        mask = (df_b["phase"] == "Fase_A_Seleccion") & (df_b["model"] == "Benchmark_Promedio")
        if mask.any():
            threshold = float(df_b.loc[mask, "mean_qlike_micro"].iloc[0])
            print(f"Umbral de cordura Fase A (Benchmark_Promedio): mean_qlike <= {threshold:.4f}")
            return threshold
    print("[Aviso] No se encontró baselines_summary.csv. Usando umbral de cordura por defecto: 2.10")
    return 2.10


def _config_signature(row: pd.Series, int_params: list, float_params: list) -> tuple:
    sig = []
    for k in int_params:
        sig.append((k, int(round(float(row[k])))))
    for k in float_params:
        sig.append((k, round(float(row[k]), 8)))
    return tuple(sig)


def select_hybrid_top4_from_sweeps(
    sanity_max_mean_qlike: float,
    n_mean: int = 2,
    n_median: int = 2,
) -> tuple[dict, pd.DataFrame]:
    selected_all = {}
    flat_rows = []

    for arch_name, info in ARCH_REGISTRY.items():
        hist_path = info["history_file"]
        model_class = info["model_class"]
        if not hist_path.exists():
            raise FileNotFoundError(
                f"No se encontró el historial de sweep para {arch_name}: {hist_path}."
            )

        df_h = pd.read_csv(hist_path)
        required = info["int_params"] + info["float_params"] + ["val_mean_qlike", "val_median_qlike"]
        missing = [c for c in required if c not in df_h.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {hist_path.name}: {missing}")

        df_h = df_h.dropna(subset=required).copy()
        df_h = df_h[np.isfinite(df_h["val_mean_qlike"]) & np.isfinite(df_h["val_median_qlike"])].copy()

        df_h["_sig"] = df_h.apply(
            lambda r: _config_signature(r, info["int_params"], info["float_params"]), axis=1
        )
        df_h = df_h.sort_values("val_mean_qlike", ascending=True).drop_duplicates(subset=["_sig"], keep="first")

        df_valid = df_h[df_h["val_mean_qlike"] <= sanity_max_mean_qlike].copy()
        if len(df_valid) < (n_mean + n_median):
            df_valid = df_h.copy()

        candidates = []
        used_sigs = set()

        # 1. Top-2 por Mean QLIKE
        df_by_mean = df_valid.sort_values(["val_mean_qlike", "val_median_qlike"], ascending=[True, True])
        for rank_idx, (_, row) in enumerate(df_by_mean.head(n_mean).iterrows(), start=1):
            used_sigs.add(row["_sig"])
            cfg = {
                "architecture": arch_name,
                "config_id": f"{arch_name}_MeanTop{rank_idx}",
                "selection_criterion": "mean_qlike",
                "selection_rank": rank_idx,
                "run_id": str(row.get("run_id", "N/A")),
                "run_name": str(row.get("run_name", "N/A")),
                "sweep_val_mean_qlike": float(row["val_mean_qlike"]),
                "sweep_val_median_qlike": float(row["val_median_qlike"]),
                "sweep_val_std_folds": float(row.get("val_std_folds", np.nan)),
                "sweep_gen_gap": float(row.get("generalization_gap", np.nan)),
            }
            for k in info["int_params"]:
                cfg[k] = int(round(float(row[k])))
            for k in info["float_params"]:
                cfg[k] = float(row[k])
            cfg["trainable_params"] = _count_trainable_params(model_class, cfg)
            candidates.append(cfg)
            flat_rows.append(cfg)

        # 2. Top-2 por Median QLIKE
        df_by_median = df_valid[~df_valid["_sig"].isin(used_sigs)].sort_values(
            ["val_median_qlike", "val_mean_qlike"], ascending=[True, True]
        )
        for rank_idx, (_, row) in enumerate(df_by_median.head(n_median).iterrows(), start=1):
            used_sigs.add(row["_sig"])
            cfg = {
                "architecture": arch_name,
                "config_id": f"{arch_name}_MedTop{rank_idx}",
                "selection_criterion": "median_qlike",
                "selection_rank": rank_idx,
                "run_id": str(row.get("run_id", "N/A")),
                "run_name": str(row.get("run_name", "N/A")),
                "sweep_val_mean_qlike": float(row["val_mean_qlike"]),
                "sweep_val_median_qlike": float(row["val_median_qlike"]),
                "sweep_val_std_folds": float(row.get("val_std_folds", np.nan)),
                "sweep_gen_gap": float(row.get("generalization_gap", np.nan)),
            }
            for k in info["int_params"]:
                cfg[k] = int(round(float(row[k])))
            for k in info["float_params"]:
                cfg[k] = float(row[k])
            cfg["trainable_params"] = _count_trainable_params(model_class, cfg)
            candidates.append(cfg)
            flat_rows.append(cfg)

        selected_all[arch_name] = candidates

    df_selected_table = pd.DataFrame(flat_rows)
    return selected_all, df_selected_table


def _log_selection_to_wandb(df_selected_table: pd.DataFrame, project: str):
    import wandb

    run = wandb.init(
        project=project,
        name="phase-a-frozen-top12-selection",
        job_type="architecture-selection",
        config={"total_candidates": len(df_selected_table), "selection_rule": "Hybrid_2x2 (2 Mean + 2 Median)"},
    )
    wandb.log({"selected_top12_architectures": wandb.Table(dataframe=df_selected_table)})
    run.finish()


# ==============================================================================
# WORKER PARALELO CON CHECKPOINTING INCREMENTAL POR CORRIDA (CANDIDATO × SEED)
# ==============================================================================
def _run_architecture_worker(
    arch_name: str,
    candidates: list,
    seeds: list,
    threads_per_worker: int = 2,
    force_retrain: bool = False,
) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

    import torch

    torch.set_num_threads(threads_per_worker)
    device_str = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else "CPU"
    )

    t_arch_start = time.time()
    tag = f"[{arch_name:6s}]"
    print(f"{tag} Worker iniciado en PID={os.getpid()} | Dispositivo: {device_str} | Checkpoints en: {RUNS_DIR}", flush=True)

    model_class = ARCH_REGISTRY[arch_name]["model_class"]
    _, X, y, dates, _ = load_processed_data_with_dates(DATA_PATH, phase="all")

    arch_individual_preds = []
    arch_individual_folds = []
    arch_candidate_ensembles = []
    arch_ensemble_variants = []
    cand_summary_rows = []
    arch_summary_rows = []
    arch_histories = {}

    seed_scores_phase_a = {c["config_id"]: [] for c in candidates}
    seed_scores_phase_b = {c["config_id"]: [] for c in candidates}

    for cand in candidates:
        cfg_id = cand["config_id"]
        crit = cand["selection_criterion"]
        cand_seed_dfs = []

        print(
            f"{tag} -- Candidato: {cfg_id} (run: {cand['run_name']} | params: {cand['trainable_params']}) --",
            flush=True,
        )

        for seed in seeds:
            run_pred_path = RUNS_DIR / f"{cfg_id}_seed_{seed}_preds.parquet"
            run_fold_path = RUNS_DIR / f"{cfg_id}_seed_{seed}_folds.csv"
            run_hist_path = RUNS_DIR / f"{cfg_id}_seed_{seed}_history.json"

            is_cached = (
                not force_retrain
                and run_pred_path.exists()
                and run_fold_path.exists()
                and run_hist_path.exists()
            )

            if is_cached:
                df_preds = pd.read_parquet(run_pred_path)
                df_metrics = pd.read_csv(run_fold_path)
                with open(run_hist_path, "r", encoding="utf-8") as f:
                    histories = json.load(f)
                status_str = "CACHED"
            else:
                t0 = time.time()
                df_metrics, df_preds, histories = run_walk_forward(
                    model_class=model_class,
                    X_raw=X,
                    y_raw=y,
                    dates=dates,
                    config=cand,
                    train_window_days=1095,
                    internal_val_days=90,
                    val_window_days=182,
                    step_days=182,
                    epochs=45,
                    patience=10,
                    seed=seed,
                )
                elapsed = time.time() - t0
                status_str = f"{elapsed:.1f}s"

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                df_preds["architecture"] = arch_name
                df_preds["config_id"] = cfg_id
                df_preds["selection_criterion"] = crit
                df_preds["seed"] = seed

                df_metrics["architecture"] = arch_name
                df_metrics["config_id"] = cfg_id
                df_metrics["selection_criterion"] = crit
                df_metrics["seed"] = seed

                # Guardado inmediato en disco al finalizar cada semilla
                df_preds.to_parquet(run_pred_path, index=False)
                df_metrics.to_csv(run_fold_path, index=False)
                with open(run_hist_path, "w", encoding="utf-8") as f:
                    json.dump(histories, f, indent=2)

            arch_individual_preds.append(df_preds)
            arch_individual_folds.append(df_metrics)
            cand_seed_dfs.append(df_preds)
            arch_histories[f"{arch_name}_{cfg_id}_seed_{seed}"] = histories

            sub_a = df_preds[df_preds["phase"] == "Fase_A_Seleccion"]
            sub_b = df_preds[df_preds["phase"] == "Fase_B_Test_Ciego"]

            mean_a = float(sub_a["qlike"].mean())
            med_a = float(sub_a["qlike"].median())
            mean_b = float(sub_b["qlike"].mean()) if len(sub_b) > 0 else np.nan
            med_b = float(sub_b["qlike"].median()) if len(sub_b) > 0 else np.nan

            seed_scores_phase_a[cfg_id].append(mean_a)
            if len(sub_b) > 0:
                seed_scores_phase_b[cfg_id].append(mean_b)

            print(
                f"{tag}    {cfg_id} | Seed {seed:4d} -> "
                f"Fase A: {mean_a:.4f} (Med {med_a:.4f}) | "
                f"Fase B: {mean_b:.4f} (Med {med_b:.4f}) [{status_str}] -> Guardado",
                flush=True,
            )

        # Deep Ensemble por candidato (5 semillas)
        df_cand_all = pd.concat(cand_seed_dfs, ignore_index=True)
        df_ens = (
            df_cand_all.groupby("Date", as_index=False)
            .agg(
                phase=("phase", "first"),
                fold=("fold", "first"),
                y_true=("y_true", "first"),
                sigma2_pred_mean=("sigma2_pred", "mean"),
                sigma2_pred_std=("sigma2_pred", "std"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df_ens["sigma2_pred_std"] = df_ens["sigma2_pred_std"].fillna(0.0)
        df_ens["architecture"] = arch_name
        df_ens["config_id"] = cfg_id
        df_ens["selection_criterion"] = crit
        df_ens["qlike_ensemble"] = compute_qlike_vector(
            df_ens["y_true"].values, df_ens["sigma2_pred_mean"].values
        )
        arch_candidate_ensembles.append(df_ens)

    df_ranks_a = pd.DataFrame(seed_scores_phase_a, index=seeds).rank(axis=1, ascending=True)
    has_phase_b = all(len(v) == len(seeds) for v in seed_scores_phase_b.values())
    df_ranks_b = (
        pd.DataFrame(seed_scores_phase_b, index=seeds).rank(axis=1, ascending=True)
        if has_phase_b
        else None
    )

    df_arch_ind = pd.concat(arch_individual_preds, ignore_index=True)
    df_arch_folds = pd.concat(arch_individual_folds, ignore_index=True)

    for cand in candidates:
        cfg_id = cand["config_id"]
        ens_cand_df = next(
            e for e in arch_candidate_ensembles if e["config_id"].iloc[0] == cfg_id
        )
        sub_ind_cand = df_arch_ind[df_arch_ind["config_id"] == cfg_id]
        full_seed_means = [
            float(sub_ind_cand[sub_ind_cand["seed"] == s]["qlike"].mean()) for s in seeds
        ]

        scopes = [
            ("Fase_A_Seleccion", ens_cand_df[ens_cand_df["phase"] == "Fase_A_Seleccion"], seed_scores_phase_a[cfg_id], df_ranks_a[cfg_id].values),
            ("Fase_B_Test_Ciego", ens_cand_df[ens_cand_df["phase"] == "Fase_B_Test_Ciego"], seed_scores_phase_b[cfg_id], df_ranks_b[cfg_id].values if df_ranks_b is not None else [np.nan]),
            ("Muestra_Completa", ens_cand_df, full_seed_means, [np.nan]),
        ]
        for scope_name, sub_ens, sc_list, rk_list in scopes:
            if sub_ens.empty:
                continue
            cand_summary_rows.append({
                "phase": scope_name,
                "architecture": arch_name,
                "config_id": cfg_id,
                "selection_criterion": cand["selection_criterion"],
                "trainable_params": cand["trainable_params"],
                "mean_seed_qlike": float(np.mean(sc_list)),
                "std_seeds": float(np.std(sc_list, ddof=1)) if len(sc_list) > 1 else 0.0,
                "ensemble_mean_qlike": float(sub_ens["qlike_ensemble"].mean()),
                "ensemble_median_qlike": float(sub_ens["qlike_ensemble"].median()),
                "mean_rank": float(np.nanmean(rk_list)),
                "std_rank": float(np.nanstd(rk_list, ddof=1)) if np.sum(~np.isnan(rk_list)) > 1 else 0.0,
                "timesteps": cand["timesteps"],
                "hidden_size": cand["hidden_size"],
                "num_layers": cand["num_layers"],
                "dropout_rate": cand["dropout_rate"],
                "learning_rate": cand["learning_rate"],
                "weight_decay": cand["weight_decay"],
                "batch_size": cand["batch_size"],
                "kernel_size": cand.get("kernel_size", np.nan),
            })

    # Ensambles consolidados por arquitectura (Hybrid_2x2, Mean_Top2, Median_Top2)
    ens_variants = {
        "Hybrid_2x2": df_arch_ind,
        "Mean_Top2": df_arch_ind[df_arch_ind["selection_criterion"] == "mean_qlike"],
        "Median_Top2": df_arch_ind[df_arch_ind["selection_criterion"] == "median_qlike"],
    }

    for ens_type, sub_pool in ens_variants.items():
        if sub_pool.empty:
            continue
        df_arch_ens = (
            sub_pool.groupby("Date", as_index=False)
            .agg(
                phase=("phase", "first"),
                fold=("fold", "first"),
                y_true=("y_true", "first"),
                sigma2_pred=("sigma2_pred", "mean"),
                sigma2_pred_std=("sigma2_pred", "std"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df_arch_ens["sigma2_pred_std"] = df_arch_ens["sigma2_pred_std"].fillna(0.0)
        df_arch_ens["architecture"] = arch_name
        df_arch_ens["ensemble_type"] = ens_type
        df_arch_ens["n_models"] = int(sub_pool.groupby("Date").size().iloc[0])
        df_arch_ens["qlike"] = compute_qlike_vector(
            df_arch_ens["y_true"].values, df_arch_ens["sigma2_pred"].values
        )
        arch_ensemble_variants.append(df_arch_ens)

        for scope_name, sub_scope in [
            ("Fase_A_Seleccion", df_arch_ens[df_arch_ens["phase"] == "Fase_A_Seleccion"]),
            ("Fase_B_Test_Ciego", df_arch_ens[df_arch_ens["phase"] == "Fase_B_Test_Ciego"]),
            ("Muestra_Completa", df_arch_ens),
        ]:
            if sub_scope.empty:
                continue
            arch_summary_rows.append({
                "phase": scope_name,
                "architecture": arch_name,
                "ensemble_type": ens_type,
                "n_models": int(df_arch_ens["n_models"].iloc[0]),
                "mean_qlike": float(sub_scope["qlike"].mean()),
                "median_qlike": float(sub_scope["qlike"].median()),
                "std_daily_qlike": float(sub_scope["qlike"].std(ddof=1)),
                "observations": int(len(sub_scope)),
            })

    df_arch_cand_ens = pd.concat(arch_candidate_ensembles, ignore_index=True)
    df_arch_ens_all = pd.concat(arch_ensemble_variants, ignore_index=True)
    df_cand_sum = pd.DataFrame(cand_summary_rows)
    df_arch_sum = pd.DataFrame(arch_summary_rows)

    df_arch_ind.to_parquet(PARTIAL_DIR / f"{arch_name}_all_seeds.parquet", index=False)
    df_arch_folds.to_csv(PARTIAL_DIR / f"{arch_name}_folds_detail.csv", index=False)
    df_arch_cand_ens.to_parquet(PARTIAL_DIR / f"{arch_name}_cand_ensembles.parquet", index=False)
    df_arch_ens_all.to_parquet(PARTIAL_DIR / f"{arch_name}_arch_ensembles.parquet", index=False)
    df_cand_sum.to_csv(PARTIAL_DIR / f"{arch_name}_cand_summary.csv", index=False)
    df_arch_sum.to_csv(PARTIAL_DIR / f"{arch_name}_arch_summary.csv", index=False)
    with open(PARTIAL_DIR / f"{arch_name}_histories.json", "w", encoding="utf-8") as f:
        json.dump(arch_histories, f, indent=2)

    total_arch_sec = time.time() - t_arch_start
    print(f"{tag} Finalizado completamente en {total_arch_sec / 60.0:.2f} min.", flush=True)
    return {"architecture": arch_name, "elapsed_sec": total_arch_sec}


# ==============================================================================
# CONSOLIDACIÓN Y AUDITORÍA DE INTEGRIDAD TEMPORAL
# ==============================================================================
def merge_and_validate_all_partials(architectures: list = None):
    if architectures is None:
        architectures = list(ARCH_REGISTRY.keys())

    df_proc, _, _, _, _ = load_processed_data_with_dates(DATA_PATH, phase="all")

    ind_list, folds_list, cand_ens_list, arch_ens_list = [], [], [], []
    cand_sum_list, arch_sum_list = [], []
    merged_histories = {}

    for arch_name in architectures:
        p_ind = PARTIAL_DIR / f"{arch_name}_all_seeds.parquet"
        p_fld = PARTIAL_DIR / f"{arch_name}_folds_detail.csv"
        p_cens = PARTIAL_DIR / f"{arch_name}_cand_ensembles.parquet"
        p_aens = PARTIAL_DIR / f"{arch_name}_arch_ensembles.parquet"
        p_csum = PARTIAL_DIR / f"{arch_name}_cand_summary.csv"
        p_asum = PARTIAL_DIR / f"{arch_name}_arch_summary.csv"
        p_hist = PARTIAL_DIR / f"{arch_name}_histories.json"

        for p in (p_ind, p_fld, p_cens, p_aens, p_csum, p_asum, p_hist):
            if not p.exists():
                raise FileNotFoundError(f"Falta archivo parcial para consolidar: {p}")

        ind_list.append(pd.read_parquet(p_ind))
        folds_list.append(pd.read_csv(p_fld))
        cand_ens_list.append(pd.read_parquet(p_cens))
        arch_ens_list.append(pd.read_parquet(p_aens))
        cand_sum_list.append(pd.read_csv(p_csum))
        arch_sum_list.append(pd.read_csv(p_asum))
        with open(p_hist, "r", encoding="utf-8") as f:
            merged_histories.update(json.load(f))

    master_individual_df = pd.concat(ind_list, ignore_index=True)
    master_individual_df["Date"] = pd.to_datetime(master_individual_df["Date"])
    master_individual_df = master_individual_df.sort_values(
        ["architecture", "config_id", "seed", "Date"]
    ).reset_index(drop=True)

    master_cand_ens_df = pd.concat(cand_ens_list, ignore_index=True)
    master_cand_ens_df["Date"] = pd.to_datetime(master_cand_ens_df["Date"])
    master_cand_ens_df = master_cand_ens_df.sort_values(
        ["architecture", "config_id", "Date"]
    ).reset_index(drop=True)

    master_arch_ens_df = pd.concat(arch_ens_list, ignore_index=True)
    master_arch_ens_df["Date"] = pd.to_datetime(master_arch_ens_df["Date"])
    master_arch_ens_df = master_arch_ens_df.sort_values(
        ["architecture", "ensemble_type", "Date"]
    ).reset_index(drop=True)

    expected_dates = None
    base_preds_path = RESULTS_DIR / "baselines_predictions.parquet"
    if base_preds_path.exists():
        df_b = pd.read_parquet(base_preds_path)
        expected_dates = pd.DatetimeIndex(
            pd.to_datetime(df_b[df_b["model"] == "Benchmark_Promedio"]["Date"]).unique()
        ).sort_values()

    validate_predictions_integrity(
        master_individual_df,
        df_proc,
        group_cols=["architecture", "config_id", "seed"],
        expected_dates=expected_dates,
    )
    validate_predictions_integrity(
        master_cand_ens_df,
        df_proc,
        group_cols=["architecture", "config_id"],
        expected_dates=expected_dates,
    )
    validate_predictions_integrity(
        master_arch_ens_df,
        df_proc,
        group_cols=["architecture", "ensemble_type"],
        expected_dates=expected_dates,
    )

    master_individual_df.to_parquet(RESULTS_DIR / "all_seeds_predictions.parquet", index=False)
    master_cand_ens_df.to_parquet(RESULTS_DIR / "candidates_ensembles_predictions.parquet", index=False)
    master_arch_ens_df.to_parquet(RESULTS_DIR / "architecture_ensembles_predictions.parquet", index=False)

    df_all_folds = pd.concat(folds_list, ignore_index=True)
    df_all_folds.to_csv(RESULTS_DIR / "all_seeds_folds_detail.csv", index=False)

    df_cand_summary = pd.concat(cand_sum_list, ignore_index=True)
    df_cand_summary.to_csv(RESULTS_DIR / "topN_multiseed_evaluation_summary.csv", index=False)

    df_arch_summary = pd.concat(arch_sum_list, ignore_index=True)
    df_arch_summary.to_csv(RESULTS_DIR / "architecture_ensembles_summary.csv", index=False)

    with open(RESULTS_DIR / "training_histories_all_models.json", "w", encoding="utf-8") as f:
        json.dump(merged_histories, f, indent=2)

    print("\n" + "=" * 105)
    print("RESUMEN CONSOLIDADO DE ENSAMBLES POR ARQUITECTURA (Hybrid_2x2 vs Mean_Top2 vs Median_Top2):")
    print("=" * 105)
    print(df_arch_summary.to_string(index=False))

    print("\nArtefactos maestros exportados exitosamente:")
    print(f"  - {RESULTS_DIR / 'all_seeds_predictions.parquet'}")
    print(f"  - {RESULTS_DIR / 'all_seeds_folds_detail.csv'}")
    print(f"  - {RESULTS_DIR / 'candidates_ensembles_predictions.parquet'}")
    print(f"  - {RESULTS_DIR / 'architecture_ensembles_predictions.parquet'}")
    print(f"  - {RESULTS_DIR / 'topN_multiseed_evaluation_summary.csv'}")
    print(f"  - {RESULTS_DIR / 'architecture_ensembles_summary.csv'}")
    print(f"  - {RESULTS_DIR / 'training_histories_all_models.json'}")


def run_multiseed_and_ensembles(
    seeds: list = None,
    target_arch: str = "all",
    workers: int = 3,
    threads_per_worker: int = 2,
    select_only: bool = False,
    merge_only: bool = False,
    force_retrain: bool = False,
    log_wandb: bool = False,
    wandb_project: str = "prediccion-varianza-btc",
):
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    if merge_only:
        merge_and_validate_all_partials()
        return

    sanity_threshold = _get_phase_a_sanity_threshold()
    selected_configs, df_selected_table = select_hybrid_top4_from_sweeps(
        sanity_max_mean_qlike=sanity_threshold
    )

    frozen_json_path = RESULTS_DIR / "selected_top4_configs_phase_a.json"
    frozen_csv_path = RESULTS_DIR / "selected_top4_configs_phase_a.csv"

    with open(frozen_json_path, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=2)
    df_selected_table.to_csv(frozen_csv_path, index=False)

    print("\n" + "=" * 105)
    print("ARQUITECTURAS PROPUESTAS Y CONGELADAS EN FASE A (REGLA HÍBRIDA 2+2: 12 CANDIDATOS)")
    print("=" * 105)
    disp_cols = [
        "config_id", "selection_criterion", "run_name", "trainable_params",
        "timesteps", "hidden_size", "num_layers", "kernel_size", "dropout_rate",
        "learning_rate", "batch_size", "sweep_val_mean_qlike", "sweep_val_median_qlike",
    ]
    disp_cols = [c for c in disp_cols if c in df_selected_table.columns]
    print(df_selected_table[disp_cols].to_string(index=False))

    if log_wandb:
        _log_selection_to_wandb(df_selected_table, project=wandb_project)

    if select_only:
        return

    archs_to_run = list(ARCH_REGISTRY.keys()) if target_arch == "all" else [target_arch]
    effective_workers = min(workers, len(archs_to_run))

    print("\n" + "=" * 105)
    print(f"ENTRENAMIENTO PARALELO EN GPU: {archs_to_run} | Workers: {effective_workers} | Semillas: {seeds}")
    print("=" * 105)

    t_global_start = time.time()

    if effective_workers == 1:
        for arch in archs_to_run:
            _run_architecture_worker(arch, selected_configs[arch], seeds, threads_per_worker, force_retrain)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_architecture_worker,
                    arch,
                    selected_configs[arch],
                    seeds,
                    threads_per_worker,
                    force_retrain,
                ): arch
                for arch in archs_to_run
            }
            for fut in as_completed(futures):
                arch = futures[fut]
                res = fut.result()
                print(f">>> Arquitectura {arch} completada ({res['elapsed_sec'] / 60.0:.2f} min).", flush=True)

    print(f"\nTiempo total: {(time.time() - t_global_start) / 60.0:.2f} min.")

    if target_arch == "all":
        merge_and_validate_all_partials(archs_to_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación Multisemilla Paralela en GPU con Checkpointing")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--arch", type=str, default="all", choices=["all", "CNN_1D", "GRU", "LSTM"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora checkpoints previos en results/partial/runs/.")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--project", type=str, default="prediccion-varianza-btc")
    args = parser.parse_args()

    run_multiseed_and_ensembles(
        seeds=args.seeds,
        target_arch=args.arch,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        select_only=args.select_only,
        merge_only=args.merge_only,
        force_retrain=args.force_retrain,
        log_wandb=args.log_wandb,
        wandb_project=args.project,
    )