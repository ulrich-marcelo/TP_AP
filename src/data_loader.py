import pandas as pd
import numpy as np

def load_processed_data(filepath="../data/processed/processed_data.csv"):
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    if 'target_r2_t1' not in df.columns:
        raise ValueError("El dataset no contiene la variable objetivo 'target_r2_t1'.")
    y = df['target_r2_t1'].values
    X = df.drop(columns=['target_r2_t1']).values
    return X, y

def load_processed_data_with_dates(filepath="../data/processed/processed_data.csv"):
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    if 'target_r2_t1' not in df.columns:
        raise ValueError("El dataset no contiene la variable objetivo 'target_r2_t1'.")
    y = df['target_r2_t1'].values
    feature_df = df.drop(columns=['target_r2_t1'])
    X = feature_df.values
    dates = df.index.values
    return df, X, y, dates, list(feature_df.columns)