import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_FEATURE_COLS = [
    "BTC_r_oc", "BTC_r_ho", "BTC_r_lo", "BTC_r_vol",
    "SPY_r_oc", "SPY_r_ho", "SPY_r_lo", "SPY_r_vol", "SPY_r_overnight",
    "VIX_log",
]
TARGET_COL = "target_r2_t1"
PHASE_A_END_DATE = "2025-12-24"


def _read_time_indexed_csv(filepath) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    df.index.name = "Date"

    if df.index.duplicated().any():
        raise ValueError(f"El archivo {path} contiene fechas duplicadas en el índice.")
    return df


def _apply_phase_filter(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    p = phase.lower().strip()
    if p in ("all", "full", "completo"):
        return df
    if p in ("phase_a", "fase_a", "selection", "seleccion"):
        return df.loc[:PHASE_A_END_DATE]
    raise ValueError(f"Filtro de fase no reconocido: '{phase}'. Usar 'all' o 'phase_a'.")


def load_processed_data(filepath="../data/processed/processed_data.csv", phase: str = "all"):
    """
    Carga la matriz de características X (10 canales) y el vector objetivo y (r_{t+1}^2).
    Con phase='phase_a', recorta estrictamente hasta PHASE_A_END_DATE (2025-12-24)
    para evitar cualquier filtración del test ciego de 2026 durante sweeps o tuning.
    """
    _, X, y, _, _ = load_processed_data_with_dates(filepath=filepath, phase=phase)
    return X, y


def load_processed_data_with_dates(
    filepath="../data/processed/processed_data.csv",
    phase: str = "all",
):
    """
    Carga el DataFrame procesado completo junto con tensores NumPy, fechas y nombres de features.
    """
    df = _read_time_indexed_csv(filepath)
    df = _apply_phase_filter(df, phase=phase)

    if TARGET_COL not in df.columns:
        raise ValueError(f"El dataset no contiene la variable objetivo '{TARGET_COL}'.")

    missing_features = [c for c in EXPECTED_FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan columnas de features esperadas: {missing_features}")

    if df.isna().any().any():
        raise ValueError("El dataset procesado contiene valores NaN.")

    feature_df = df[EXPECTED_FEATURE_COLS]
    X = feature_df.values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    dates = df.index.values

    return df, X, y, dates, list(feature_df.columns)


def load_returns_reference(
    processed_filepath="../data/processed/processed_data.csv",
    ref_filepath=None,
    raw_filepath=None,
    phase: str = "all",
) -> pd.DataFrame:
    """
    Carga la serie exacta de retornos close-to-close (BTC_r_cc) y target_r2_t1
    alineada estrictamente al índice de processed_data.csv para modelos econométricos
    (EGARCH / EGARCH-X) y diagnóstico de residuos estandarizados.
    """
    proc_path = Path(processed_filepath)
    df_proc = _apply_phase_filter(_read_time_indexed_csv(proc_path), phase=phase)

    if ref_filepath is None:
        ref_filepath = proc_path.parent / "returns_reference.csv"
    else:
        ref_filepath = Path(ref_filepath)

    if ref_filepath.exists():
        df_ref = _read_time_indexed_csv(ref_filepath).reindex(df_proc.index)
    else:
        if raw_filepath is None:
            raw_filepath = proc_path.parents[1] / "raw" / "raw_data.csv"
        df_raw = _read_time_indexed_csv(raw_filepath)
        btc_r_cc = np.log(df_raw["BTC_Close"] / df_raw["BTC_Close"].shift(1))
        df_ref = pd.DataFrame({
            "BTC_r_cc": btc_r_cc.reindex(df_proc.index),
            TARGET_COL: df_proc[TARGET_COL],
        }, index=df_proc.index)
        df_ref.index.name = "Date"

    if not df_ref.index.equals(df_proc.index):
        raise ValueError("El índice de returns_reference no coincide exactamente con processed_data.")

    if df_ref[["BTC_r_cc", TARGET_COL]].isna().any().any():
        raise ValueError("Existen NaNs en returns_reference.")

    # Verificación estricta de causalidad temporal: target_t == (BTC_r_cc_{t+1})^2
    if not np.allclose(
        df_proc[TARGET_COL].iloc[:-1].values,
        (df_ref["BTC_r_cc"].iloc[1:].values ** 2),
        rtol=1e-10,
        atol=1e-14,
    ):
        raise ValueError("Desalineación temporal detectada entre BTC_r_cc y target_r2_t1.")

    return df_ref


def load_dataset_metadata(processed_filepath="../data/processed/processed_data.csv") -> dict:
    """Carga los metadatos JSON generados por 1_data_builder.py si están disponibles."""
    meta_path = Path(processed_filepath).parent / "dataset_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)