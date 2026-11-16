import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==============================================================================
# 1. LISTA DE FECHAS DE OUTLIERS A EXCLUIR
# ==============================================================================
# Podés agregar, quitar o editar fechas en formato 'YYYY-MM-DD'.
# Estas fechas se enmascaran (NaN) para que las curvas no se distorsionen
# y en su lugar se inserta una línea vertical roja punteada.
OUTLIER_DATES = [
   # '2020-03-12',  # Crash de liquidez COVID-19 (Black Thursday)
    #'2020-03-13',  # Continuación del shock de volatilidad extrema
    '2020-03-11',  # Gran liquidación masiva del mercado cripto
]

# ==============================================================================
# 2. FUNCIÓN DE CÁLCULO QLIKE
# ==============================================================================
def compute_qlike_vector(y_true, sigma2_pred, eps=1e-6):
    """Calcula la pérdida QLIKE diaria de forma numéricamente estable."""
    y = np.clip(y_true, eps, None)
    s2 = np.clip(sigma2_pred, eps, None)
    ratio = y / s2
    return ratio - np.log(ratio) - 1.0

# ==============================================================================
# 3. FUNCIÓN DE GENERACIÓN DE GRÁFICOS
# ==============================================================================
def plot_architecture_results(
    parquet_path: str,
    arch_name: str,
    outlier_dates: list = OUTLIER_DATES,
    selected_config_id: str = None,
    use_ensemble: bool = True,
    seed: int = 42,
    output_dir: str = "./figures"
):
    """
    Carga las predicciones de una arquitectura, calcula QLIKE y genera dos gráficos:
      1. Valor Real vs Predicción temporal.
      2. Pérdida QLIKE a lo largo del tiempo.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Si hay varias configuraciones TopN, tomamos la primera o la indicada
    if selected_config_id is None:
        selected_config_id = df['config_id'].iloc[0]
    df_config = df[df['config_id'] == selected_config_id].copy()

    # Agregación: Ensamble (promedio de semillas) o semilla particular
    if use_ensemble:
        df_plot = df_config.groupby('Date', as_index=False).agg({
            'y_true': 'first',
            'sigma2_pred': 'mean'
        }).sort_values('Date').reset_index(drop=True)
        model_label = f"{arch_name} (Ensemble Top-1)"
    else:
        df_plot = df_config[df_config['seed'] == seed].sort_values('Date').reset_index(drop=True)
        model_label = f"{arch_name} (Seed {seed})"

    # Cálculo del QLIKE diario
    df_plot['qlike'] = compute_qlike_vector(
        df_plot['y_true'].values, 
        df_plot['sigma2_pred'].values
    )

    # Identificación y enmascaramiento de outliers para preservar la escala
    outlier_dt = pd.to_datetime(outlier_dates)
    is_outlier = df_plot['Date'].isin(outlier_dt)

    df_clean = df_plot.copy()
    df_clean.loc[is_outlier, ['y_true', 'sigma2_pred', 'qlike']] = np.nan

    outliers_presentes = df_plot[is_outlier]['Date'].tolist()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estilo visual
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --------------------------------------------------------------------------
    # GRÁFICO 1: Valor Real vs Predicción
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)

    ax.plot(df_clean['Date'], df_clean['y_true'], 
            label='Valor Real ($y_t$)', color='#7f7f7f', alpha=0.6, linewidth=1.1)
    ax.plot(df_clean['Date'], df_clean['sigma2_pred'], 
            label=f'Predicción $\\hat{{\\sigma}}^2_t$ ({model_label})', color='#1f77b4', linewidth=1.4)

    # Dibujar líneas verticales rojas en las fechas outliers
    for i, date in enumerate(outliers_presentes):
        ax.axvline(x=date, color='#d62728', linestyle='--', linewidth=1.2, 
                   label='Outlier Excluido' if i == 0 else "")

    ax.set_title(f"{arch_name}: Valor Real vs. Predicción", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Varianza / Volatilidad", fontsize=11)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper right', frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path_val = out_dir / f"{arch_name}_real_vs_pred.png"
    fig.savefig(save_path_val)
    plt.close(fig)
    print(f"Guardado: {save_path_val}")

    # --------------------------------------------------------------------------
    # GRÁFICO 2: Pérdida QLIKE a lo largo del tiempo
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)

    ax.plot(df_clean['Date'], df_clean['qlike'], 
            label=f'QLIKE ({model_label})', color='#2ca02c', linewidth=1.2)

    for i, date in enumerate(outliers_presentes):
        ax.axvline(x=date, color='#d62728', linestyle='--', linewidth=1.2, 
                   label='Outlier Excluido' if i == 0 else "")

    ax.set_title(f"{arch_name}: Función de Pérdida QLIKE en el Tiempo", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("QLIKE Loss", fontsize=11)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper right', frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path_qlike = out_dir / f"{arch_name}_qlike_timeline.png"
    fig.savefig(save_path_qlike)
    plt.close(fig)
    print(f"Guardado: {save_path_qlike}")


# ==============================================================================
# 4. EJECUCIÓN PARA LAS TRES ARQUITECTURAS
# ==============================================================================
if __name__ == '__main__':
    architectures = {
        'CNN_1D': 'results/CNN_1D_topN_predictions.parquet',
        'GRU':    'results/GRU_topN_predictions.parquet',
        'LSTM':   'results/LSTM_topN_predictions.parquet'
    }

    for arch, file_path in architectures.items():
        print(f"\nProcesando arquitectura: {arch}...")
        try:
            plot_architecture_results(
                parquet_path=file_path,
                arch_name=arch,
                outlier_dates=OUTLIER_DATES,
                use_ensemble=True,
                output_dir="./graficos_resultados"
            )
        except Exception as e:
            print(f"Error procesando {arch}: {e}")