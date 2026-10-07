import pandas as pd
import numpy as np

def load_processed_data(filepath="../data/processed/processed_data.csv"):
    """
    Carga el dataset procesado, aislando el target de las variables predictoras.
    """
    # Parseamos las fechas y las asignamos como índice para mantener la noción temporal
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    
    # Verificación de integridad metodológica
    if 'target_r2_t1' not in df.columns:
        raise ValueError("El dataset no contiene la variable objetivo 'target_r2_t1'.")
    
    # Extraemos el target
    y = df['target_r2_t1'].values
    
    # El resto de las columnas son nuestros features (los 10 canales multivariados)
    X = df.drop(columns=['target_r2_t1']).values
    
    return X, y