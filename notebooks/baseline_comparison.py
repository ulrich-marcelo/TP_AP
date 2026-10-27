import sys
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_processed_data_with_dates

def run_walk_forward_egarch_benchmark():
    df, X, y, dates, feature_names = load_processed_data_with_dates(ROOT / "data" / "processed" / "processed_data.csv")
    
    # El retorno logarítmico diario en porcentaje acelera la convergencia de arch_model
    # r_BTC = ln(C_t / C_t-1)
    # y = r_{t+1}^2
    returns = np.sign(df['BTC_r_oc'].values) * np.sqrt(df['target_r2_t1'].values) * 100.0
    
    train_window_days = 1095
    val_window_days = 182
    step_days = 182
    total_days = len(returns)

    results_baseline = []
    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_days:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_days)

        if (val_end - val_start) < 90:
            break

        r_train = returns[train_start:train_end]
        r_val = returns[val_start:val_end]
        y_val_raw = y[val_start:val_end]
        dates_val = dates[val_start:val_end]
        
        # Variables exógenas normalizadas para el EGARCH-X
        X_train_exog = X[train_start:train_end, [3, 4, 9]] # ej. BTC_vol, SPY_ret, VIX_log
        X_val_exog = X[val_start:val_end, [3, 4, 9]]

        # Especificación: EGARCH(1,3) con distribución GED
        am = arch_model(
            r_train, 
            mean='Constant', 
            vol='EGARCH', 
            p=1, 
            o=1, 
            q=3, 
            dist='ged',
            x=X_train_exog # Omitir este argumento para el EGARCH univariado estándar
        )
        res = am.fit(disp='off', show_warning=False)

        # Pronóstico fuera de muestra a 1 paso
        forecasts = res.forecast(horizon=1, start=0)
        # Convertir varianza de retornos porcentuales a escala decimal: / 10000.0
        # En rolling out-of-sample estricto:
        pred_sigma2_fold = []
        for t in range(len(r_val)):
            # Estimación recursiva o evaluación condicional
            pred_sigma2_fold.append((res.params['omega'] / 10000.0)) # simplificado para el fold

        start_idx += step_days
        fold_idx += 1