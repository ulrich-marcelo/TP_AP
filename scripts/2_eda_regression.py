import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_processed_data_with_dates

def run_linear_diagnostics():
    df, X, y, dates, feature_names = load_processed_data_with_dates(ROOT / "data" / "processed" / "processed_data.csv")
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CORRELACIÓN Y REGRESIÓN LINEAL (BASELINE MCO)")
    print("="*70)
    
    # 1. Matriz de Correlación de Pearson con el Target
    corr_series = pd.Series(index=feature_names, dtype=float)
    for i, col in enumerate(feature_names):
        corr_series[col] = np.corrcoef(X[:, i], y)[0, 1]
    
    print("\nCorrelación de Pearson respecto a target_r2_t1:")
    print(corr_series.to_string())

    # 2. Diagnóstico de Multicolinealidad (VIF)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    print("\nFactor de Inflación de la Varianza (VIF):")
    print(vif_data.to_string(index=False))

    # 3. Regresión Lineal OLS
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    print("\nResumen Estadístico OLS (Target: r_{BTC, t+1}^2):")
    print(ols_model.summary(xname=["const"] + feature_names))
    
    return ols_model

if __name__ == "__main__":
    run_linear_diagnostics()