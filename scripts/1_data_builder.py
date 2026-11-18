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


def build_datasets(start_date="2016-01-01", end_date="2026-01-01"):
    tickers = {"BTC": "BTC-USD", "SPY": "SPY", "VIX": "^VIX"}
    raw_data = {}

    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        raw_data[name] = df

    # Guardar respaldo crudo alineado al calendario continuo de BTC
    df_raw = pd.concat([df.add_prefix(f"{k}_") for k, df in raw_data.items()], axis=1)
    df_raw = df_raw.dropna(subset=["BTC_Close"])
    df_raw.to_csv(RAW_DIR / "raw_data.csv")

    # 1. Canales Endógenos de Bitcoin (calendario continuo 24/7)
    btc = raw_data["BTC"].copy()
    dataset = pd.DataFrame(index=btc.index)
    dataset["BTC_r_oc"] = np.log(btc["Close"] / btc["Open"])
    dataset["BTC_r_ho"] = np.log(btc["High"] / btc["Open"])
    dataset["BTC_r_lo"] = np.log(btc["Low"] / btc["Open"])

    vol_btc = btc["Volume"].replace(0, np.nan).ffill()
    dataset["BTC_r_vol"] = np.log(vol_btc / vol_btc.shift(1))

    # Variable objetivo: varianza realizada proxied por r_{t+1}^2
    btc_r_close = np.log(btc["Close"] / btc["Close"].shift(1))
    dataset["target_r2_t1"] = (btc_r_close ** 2).shift(-1)

    # 2. Canales Exógenos S&P 500 (calculados SOLO en días hábiles antes del join)
    spy = raw_data["SPY"].copy()
    spy["SPY_r_oc"] = np.log(spy["Close"] / spy["Open"])
    spy["SPY_r_ho"] = np.log(spy["High"] / spy["Open"])
    spy["SPY_r_lo"] = np.log(spy["Low"] / spy["Open"])

    vol_spy = spy["Volume"].replace(0, np.nan).ffill()
    spy["SPY_r_vol"] = np.log(vol_spy / vol_spy.shift(1))
    # El lunes compara Open_lunes vs Close_viernes automáticamente
    spy["SPY_r_overnight"] = np.log(spy["Open"] / spy["Close"].shift(1))

    spy_cols = ["SPY_r_oc", "SPY_r_ho", "SPY_r_lo", "SPY_r_vol", "SPY_r_overnight"]
    dataset = dataset.join(spy[spy_cols])
    # Sábados, domingos y feriados bursátiles = 0.0 estricto en retornos/flujos
    dataset[spy_cols] = dataset[spy_cols].fillna(0.0)

    # 3. Canal Exógeno de Estado VIX (ffill mantiene nivel vigente del viernes)
    vix = raw_data["VIX"].copy()
    vix["VIX_log"] = np.log(vix["Close"])
    dataset = dataset.join(vix[["VIX_log"]])
    dataset["VIX_log"] = dataset["VIX_log"].ffill()

    # Eliminar bordes iniciales/finales con NaN
    dataset = dataset.dropna()

    # Orden canónico de columnas
    ordered_cols = [
        "BTC_r_oc", "BTC_r_ho", "BTC_r_lo", "BTC_r_vol",
        "target_r2_t1",
        "SPY_r_oc", "SPY_r_ho", "SPY_r_lo", "SPY_r_vol", "SPY_r_overnight",
        "VIX_log",
    ]
    dataset = dataset[ordered_cols]

    # --- SANITY CHECKS AUTOMÁTICOS ---
    assert not dataset.isna().any().any(), "ERROR: Existen NaNs en el dataset procesado."
    assert np.isfinite(dataset.values).all(), "ERROR: Existen valores infinitos en el dataset."
    weekends = dataset.index.dayofweek >= 5
    assert (dataset.loc[weekends, spy_cols] == 0.0).all().all(), "ERROR: SPY no es 0.0 en fines de semana."
    assert (dataset.loc[weekends, "VIX_log"] > 0.0).all(), "ERROR: VIX_log inválido en fines de semana."

    out_path = PROC_DIR / "processed_data.csv"
    dataset.to_csv(out_path)
    print(f"Dataset guardado exitosamente en: {out_path} | Shape: {dataset.shape}")
    print("\nVerificación Viernes-Lunes (Enero 2020):")
    print(dataset.loc["2020-01-03":"2020-01-06", spy_cols + ["VIX_log"]])
    return dataset


if __name__ == "__main__":
    build_datasets()