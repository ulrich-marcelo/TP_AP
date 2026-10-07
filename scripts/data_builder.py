import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_calculate_features(start_date="2016-01-01", end_date="2026-01-01"):
    # 1. Descarga de datos
    tickers = {"BTC": "BTC-USD", "SPY": "SPY", "VIX": "^VIX"}
    data = {}
    
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Eliminar MultiIndex de las columnas si yfinance lo devuelve
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        data[name] = df

    # 2. Variables Endógenas de Bitcoin (4 canales)
    btc = data["BTC"].copy()
    btc['BTC_r_oc'] = np.log(btc['Close'] / btc['Open'])
    btc['BTC_r_ho'] = np.log(btc['High'] / btc['Open'])
    btc['BTC_r_lo'] = np.log(btc['Low'] / btc['Open'])
    btc['BTC_r_vol'] = np.log(btc['Volume'] / btc['Volume'].shift(1))
    
    # Target: Retorno logarítmico de cierre t, al cuadrado, desplazado a t+1
    # r_BTC,t = ln(C_t / C_t-1)
    btc['BTC_r_close'] = np.log(btc['Close'] / btc['Close'].shift(1))
    btc['target_r2_t1'] = (btc['BTC_r_close'] ** 2).shift(-1)
    
    # 3. Variables Exógenas del S&P 500 (5 canales)
    spy = data["SPY"].copy()
    spy['SPY_r_oc'] = np.log(spy['Close'] / spy['Open'])
    spy['SPY_r_ho'] = np.log(spy['High'] / spy['Open'])
    spy['SPY_r_lo'] = np.log(spy['Low'] / spy['Open'])
    spy['SPY_r_vol'] = np.log(spy['Volume'] / spy['Volume'].shift(1))
    # Retorno Overnight: ln(O_t / C_t-1)
    spy['SPY_r_overnight'] = np.log(spy['Open'] / spy['Close'].shift(1))
    
    # 4. Variable Exógena VIX (1 canal)
    vix = data["VIX"].copy()
    vix['VIX_log'] = np.log(vix['Close'])
    
    # 5. Fusión y Alineación Temporal (Manejo de Fines de Semana)
    # Mantener el índice continuo de BTC
    dataset = btc[['BTC_r_oc', 'BTC_r_ho', 'BTC_r_lo', 'BTC_r_vol', 'target_r2_t1']].copy()
    
    # Unir SPY y VIX
    dataset = dataset.join(spy[['SPY_r_oc', 'SPY_r_ho', 'SPY_r_lo', 'SPY_r_vol', 'SPY_r_overnight']])
    dataset = dataset.join(vix[['VIX_log']])
    
    # Propagar el último valor conocido de SPY y VIX durante sábados y domingos
    dataset = dataset.ffill()
    
    # Eliminar las filas iniciales con NaNs producidas por el shift(1)
    dataset = dataset.dropna()
    
    return dataset

# Ejecución
df_final = fetch_and_calculate_features()

print(df_final.shape)