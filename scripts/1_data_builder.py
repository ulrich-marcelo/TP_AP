import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

PHASE_A_END_DATE = "2025-12-25"


def _download_clean_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga OHLCV desde yfinance, elimina duplicados y normaliza índice sin zona horaria."""
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Descarga vacía desde yfinance para el ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df.index.name = "Date"
    return df


def build_datasets(start_date: str = "2016-01-01", end_date: str = "2026-11-01") -> pd.DataFrame:
    """
    Construye el dataset procesado hasta el 2026-10-30 (end_date exclusivo en yfinance = 2026-11-01).
    """
    tickers = {"BTC": "BTC-USD", "SPY": "SPY", "VIX": "^VIX"}
    raw_data = {
        name: _download_clean_ohlcv(ticker, start_date, end_date)
        for name, ticker in tickers.items()
    }

    # -------------------------------------------------------------------------
    # 0. Respaldo crudo alineado al calendario continuo 24/7 de BTC (sin ffill)
    # -------------------------------------------------------------------------
    btc = raw_data["BTC"]
    df_raw = pd.concat([df.add_prefix(f"{k}_") for k, df in raw_data.items()], axis=1)
    df_raw = df_raw.loc[btc.index].sort_index()
    df_raw.index.name = "Date"
    raw_out_path = RAW_DIR / "raw_data.csv"
    df_raw.to_csv(raw_out_path, index_label="Date")

    # -------------------------------------------------------------------------
    # 1. Canales Endógenos de Bitcoin (calendario continuo 24/7)
    # -------------------------------------------------------------------------
    dataset = pd.DataFrame(index=btc.index)
    dataset.index.name = "Date"

    dataset["BTC_r_oc"] = np.log(btc["Close"] / btc["Open"])
    dataset["BTC_r_ho"] = np.log(btc["High"] / btc["Open"])
    dataset["BTC_r_lo"] = np.log(btc["Low"] / btc["Open"])

    vol_btc = btc["Volume"].replace(0, np.nan).ffill()
    dataset["BTC_r_vol"] = np.log(vol_btc / vol_btc.shift(1))

    # Retorno close-to-close en t y variable objetivo r_{BTC, t+1}^2
    btc_r_close = np.log(btc["Close"] / btc["Close"].shift(1))
    dataset["target_r2_t1"] = (btc_r_close ** 2).shift(-1)

    # -------------------------------------------------------------------------
    # 2. Canales Exógenos S&P 500 (calculados SOLO en días hábiles antes del join)
    # -------------------------------------------------------------------------
    spy = raw_data["SPY"].copy()
    spy["SPY_r_oc"] = np.log(spy["Close"] / spy["Open"])
    spy["SPY_r_ho"] = np.log(spy["High"] / spy["Open"])
    spy["SPY_r_lo"] = np.log(spy["Low"] / spy["Open"])

    vol_spy = spy["Volume"].replace(0, np.nan).ffill()
    spy["SPY_r_vol"] = np.log(vol_spy / vol_spy.shift(1))
    # El lunes (o día post-feriado) compara Open_t vs Close_{último hábil}
    spy["SPY_r_overnight"] = np.log(spy["Open"] / spy["Close"].shift(1))

    spy_cols = ["SPY_r_oc", "SPY_r_ho", "SPY_r_lo", "SPY_r_vol", "SPY_r_overnight"]
    dataset = dataset.join(spy[spy_cols], how="left")
    # Sábados, domingos y feriados bursátiles = 0.0 estricto en retornos y volumen
    dataset[spy_cols] = dataset[spy_cols].fillna(0.0)

    # -------------------------------------------------------------------------
    # 3. Canal Exógeno de Estado VIX (ffill mantiene el nivel del último cierre)
    # -------------------------------------------------------------------------
    vix = raw_data["VIX"].copy()
    vix["VIX_log"] = np.log(vix["Close"])
    dataset = dataset.join(vix[["VIX_log"]], how="left")
    dataset["VIX_log"] = dataset["VIX_log"].ffill()

    # Eliminar bordes iniciales (primer día sin lag) y final (último día sin t+1)
    dataset = dataset.dropna().sort_index()
    dataset.index.name = "Date"

    # Orden canónico estricto: 10 features + 1 target
    feature_cols = [
        "BTC_r_oc", "BTC_r_ho", "BTC_r_lo", "BTC_r_vol",
        "SPY_r_oc", "SPY_r_ho", "SPY_r_lo", "SPY_r_vol", "SPY_r_overnight",
        "VIX_log",
    ]
    ordered_cols = feature_cols + ["target_r2_t1"]
    dataset = dataset[ordered_cols]

    # Serie auxiliar de retornos close-to-close alineada al mismo índice (para EGARCH y diagnósticos)
    ref_df = pd.DataFrame({
        "BTC_r_cc": btc_r_close.loc[dataset.index],
        "target_r2_t1": dataset["target_r2_t1"],
    }, index=dataset.index)
    ref_df.index.name = "Date"

    # -------------------------------------------------------------------------
    # 4. Sanity Checks Automáticos de Integridad Temporal y Econométrica
    # -------------------------------------------------------------------------
    assert not dataset.isna().any().any(), "ERROR: Existen NaNs en el dataset procesado."
    assert not ref_df.isna().any().any(), "ERROR: Existen NaNs en returns_reference."
    assert np.isfinite(dataset.values).all(), "ERROR: Existen valores infinitos en el dataset."
    assert dataset.index.is_monotonic_increasing, "ERROR: El índice temporal no es creciente."
    assert not dataset.index.duplicated().any(), "ERROR: Existen fechas duplicadas en el índice."

    non_trading_days = df_raw.loc[dataset.index, "SPY_Close"].isna()
    assert (dataset.loc[non_trading_days, spy_cols] == 0.0).all().all(), (
        "ERROR: Existen retornos/flujos SPY distintos de 0.0 en días sin rueda bursátil."
    )
    assert (dataset.loc[non_trading_days, "VIX_log"] > 0.0).all(), (
        "ERROR: VIX_log inválido en días sin rueda bursátil."
    )
    assert np.allclose(
        dataset["target_r2_t1"].iloc[:-1].values,
        (ref_df["BTC_r_cc"].iloc[1:].values ** 2),
        rtol=1e-12,
        atol=1e-15,
    ), "ERROR: Desalineación temporal entre BTC_r_cc en t+1 y target_r2_t1 en t."

    # -------------------------------------------------------------------------
    # 5. Persistencia
    # -------------------------------------------------------------------------
    out_path = PROC_DIR / "processed_data.csv"
    ref_path = PROC_DIR / "returns_reference.csv"
    meta_path = PROC_DIR / "dataset_metadata.json"

    dataset.to_csv(out_path, index_label="Date")
    ref_df.to_csv(ref_path, index_label="Date")

    phase_a_df = dataset.loc[:PHASE_A_END_DATE]
    phase_b_df = dataset.loc[pd.Timestamp(PHASE_A_END_DATE) + pd.Timedelta(days=1) :]

    metadata = {
        "start_date_requested": start_date,
        "end_date_requested": end_date,
        "actual_start_date": str(dataset.index.min().date()),
        "actual_end_date": str(dataset.index.max().date()),
        "phase_a_end_date": PHASE_A_END_DATE,
        "total_observations": int(len(dataset)),
        "phase_a_observations": int(len(phase_a_df)),
        "phase_b_observations": int(len(phase_b_df)),
        "us_trading_days": int((~non_trading_days).sum()),
        "us_non_trading_days": int(non_trading_days.sum()),
        "feature_columns": feature_cols,
        "target_column": "target_r2_t1",
        "target_mean": float(dataset["target_r2_t1"].mean()),
        "target_median": float(dataset["target_r2_t1"].median()),
        "target_max": float(dataset["target_r2_t1"].max()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset crudo guardado en:      {raw_out_path} | Shape: {df_raw.shape}")
    print(f"Dataset procesado guardado en:  {out_path} | Shape: {dataset.shape}")
    print(f"Referencia retornos guardada:   {ref_path} | Shape: {ref_df.shape}")
    print(f"Metadatos guardados en:         {meta_path}")
    print(f"Rango total: {metadata['actual_start_date']} -> {metadata['actual_end_date']} "
          f"(Fase A <= {PHASE_A_END_DATE}: {metadata['phase_a_observations']} obs | "
          f"Fase B Ciego 2026: {metadata['phase_b_observations']} obs)")
    return dataset


if __name__ == "__main__":
    build_datasets()