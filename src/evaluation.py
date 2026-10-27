import numpy as np
import pandas as pd
from scipy import stats

def diebold_mariano_test(loss_baseline, loss_candidate, h=1):
    """
    Test de Diebold-Mariano (1995) con corrección HAC (Newey-West)
    y ajuste de muestra finita de Harvey, Leybourne y Newbold (1997).
    
    loss_baseline: Vector 1D o Serie con errores QLIKE del modelo base (EGARCH).
    loss_candidate: Vector 1D o Serie con errores QLIKE del modelo candidato (DL).
    h: Horizonte temporal de predicción (para t+1, h=1).
    """
    d = np.array(loss_baseline, dtype=np.float64) - np.array(loss_candidate, dtype=np.float64)
    T = len(d)
    if T == 0:
        raise ValueError("Los vectores de pérdida no pueden estar vacíos.")
        
    mean_d = np.mean(d)
    
    # Estimador HAC con ponderador de Bartlett
    gamma = []
    for k in range(h):
        if k == 0:
            gamma.append(np.var(d, ddof=0))
        else:
            cov_k = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
            gamma.append(cov_k)
            
    var_d = gamma[0] + 2.0 * sum(
        (1.0 - k / (h + 1.0)) * gamma[k] for k in range(1, h)
    )
    
    if var_d <= 0.0:
        var_d = np.var(d, ddof=1)
        
    se_d = np.sqrt(var_d / T)
    if se_d == 0.0:
        return 0.0, 1.0
        
    dm_stat = mean_d / se_d
    
    # Corrección de Harvey-Leybourne-Newbold
    hln_adj = np.sqrt((T + 1.0 - 2.0 * h + (h / T) * (h - 1.0)) / T)
    dm_stat_corrected = dm_stat * hln_adj
    
    # p-valor bilateral bajo distribución t-Student con T-1 grados de libertad
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat_corrected), df=T - 1))
    
    return float(dm_stat_corrected), float(p_value)

def analyze_covid_regime(df_preds):
    """
    Filtra y analiza el desempeño del modelo en la ventana del colapso de marzo 2020.
    """
    df = df_preds.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= '2020-02-15') & (df['Date'] <= '2020-05-30')
    df_covid = df[mask].copy()
    
    return {
        'num_samples': len(df_covid),
        'mean_qlike': float(df_covid['qlike'].mean()) if len(df_covid) > 0 else np.nan,
        'median_qlike': float(df_covid['qlike'].median()) if len(df_covid) > 0 else np.nan,
        'std_qlike': float(df_covid['qlike'].std()) if len(df_covid) > 0 else np.nan,
        'max_y_true': float(df_covid['y_true'].max()) if len(df_covid) > 0 else np.nan,
        'max_sigma2_pred': float(df_covid['sigma2_pred'].max()) if len(df_covid) > 0 else np.nan,
        'subset_df': df_covid
    }