# Modelado y Predicción de la Varianza Condicional Diaria de Bitcoin (2016–2026)

**Un enfoque comparativo entre Arquitecturas de Aprendizaje Profundo y Econometría Clásica**

Este repositorio contiene el código fuente, los pipelines de preprocesamiento, la configuración experimental y los artefactos reproducibles del Trabajo Práctico Final de la materia **Aprendizaje Profundo** (Licenciatura en Ciencia de Datos, ECyT - UNSAM).

El proyecto modela y evalúa fuera de muestra la varianza condicional diaria de Bitcoin (`BTC`) durante el período **2016–2026** ($3.953$ observaciones diarias), utilizando el **retorno logarítmico diario al cuadrado ($r_{\text{BTC}, t+1}^2$)** como proxy empírico no sesgado de la volatilidad latente. Se contrastan tres familias de redes neuronales profundas (**CNN 1D**, **GRU** y **LSTM**) alimentadas con un tensor estacionario de 10 canales endógenos y exógenos frente a tres modelos de referencia:

* `Benchmark_Promedio`: varianza incondicional móvil sobre la ventana de entrenamiento (modelo nulo).
* `EGARCH(1,3)_GED`: modelo econométrico heterocedástico univariado con errores GED.
* `EGARCH(2,3)-X_GED`: extensión econométrica multivariada con covariables exógenas en la ecuación de log-varianza.

---

## 1. Estructura del Repositorio

```text
├── data/
│   ├── raw/
│   │   └── raw_data.csv                           # Series históricas OHLCV descargadas (BTC-USD, SPY, ^VIX)
│   └── processed/
│       ├── processed_data.csv                     # Tensor estacionario de 10 canales + target_r2_t1 (3.953 filas)
│       ├── returns_reference.csv                  # Retornos exactos close-to-close (BTC_r_cc) para EGARCH y residuos
│       └── dataset_metadata.json                  # Metadatos de cobertura temporal y alineación de calendarios
├── notebooks/
│   └── baseline.ipynb                             # Exploración inicial de modelos de referencia
├── scripts/
│   ├── 1_data_builder.py                          # Descarga desde Yahoo Finance, alineación de calendarios y features
│   ├── 2_eda_regression.py                        # EDA en Fase A: ADF/KPSS, ACF, Pearson/Spearman/IM, VIF, OLS (HC3)
│   ├── 3_tune_egarch_x.py                         # Búsqueda en grilla de EGARCH-X en Fase A (minimización de QLIKE)
│   ├── 4_evaluate_baselines.py                    # Evaluación Walk-Forward (16 folds) de los 3 modelos de referencia
│   ├── 5_run_sweep_cnn.py                         # Sweep bayesiano en W&B para CNN 1D (40 corridas, solo Fase A)
│   ├── 6_run_sweep_gru.py                         # Sweep bayesiano en W&B para GRU (40 corridas, solo Fase A)
│   ├── 7_run_sweep_lstm.py                        # Sweep bayesiano en W&B para LSTM (40 corridas, solo Fase A)
│   ├── 8_run_experiments_seeds.py                 # Selección dual 2+2, reentrenamiento en 5 semillas y ensambles
│   ├── 9_compare_dl_vs_baselines.py               # Métricas consolidadas, Diebold-Mariano (HLN + HAC NW) y gráficos
│   ├── 10_error_analysis_and_diagnostics.py       # Análisis por regímenes, Top-15 shocks y residuos estandarizados
│   ├── 11_prediction_plots.py                     # Generación de figuras individuales de publicación (300 DPI)
│   └── baseline_egarch.py                         # Script auxiliar de inspección de parámetros de EGARCH(1,3)_GED
├── src/
│   ├── __init__.py                                # API pública del paquete src
│   ├── data_loader.py                             # Carga validada de tensores, retornos y corte estricto Fase A/B
│   ├── econometrics.py                            # Implementación MLE y pronóstico recursivo h=1 de EGARCH-X
│   ├── evaluation.py                              # QLIKE vectorizado, Diebold-Mariano (HLN/HAC) y contrato temporal
│   ├── models.py                                  # Arquitecturas PyTorch: VolatilidadCNN, VolatilidadGRU, VolatilidadLSTM
│   └── trainer.py                                 # QLIKELoss en log-varianza, Walk-Forward causal y Early Stopping
├── results/
│   ├── eda/                                       # Tablas CSV del diagnóstico exploratorio en Fase A
│   ├── sweeps/                                    # Respaldos locales de las 120 corridas bayesianas de W&B
│   ├── figures/                                   # Gráficos comparativos y subcarpeta report_individual/ (00 a 08)
│   ├── selected_top4_configs_phase_a.{csv,json}   # 12 configuraciones congeladas al cierre de Fase A
│   ├── *.parquet                                  # Predicciones diarias alineadas (baselines, semillas y ensambles)
│   └── *.csv / *.json                             # Resúmenes QLIKE, tests DM/HAC, regímenes, residuos e historiales
├── requirements.txt                               # Dependencias exactas del entorno virtual
└── README.md                                      # Documentación del proyecto
```

---

## 2. Instalación y Configuración del Entorno

### 2.1 Clonar el repositorio y crear entorno virtual

```bash
git clone [https://github.com/ulrich-marcelo/tp_ap.git](https://github.com/ulrich-marcelo/tp_ap.git)
cd tp_ap

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .\.venv\Scripts\activate       # Windows
```

### 2.2 Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2.3 Aceleración por Hardware (CUDA / GPU)

Para entrenar los *sweeps* y el experimento multisemilla en GPU NVIDIA, verificar previamente el driver (`nvidia-smi`) e instalar la rueda oficial de PyTorch correspondiente a la versión de CUDA instalada desde [pytorch.org](https://pytorch.org/get-started/locally/). Comprobación rápida:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

El módulo `src/trainer.py` detecta automáticamente `torch.cuda.is_available()`, fijando semillas determinísticas y habilitando `pin_memory=True` y transferencias `non_blocking=True`. Los scripts de EDA, baselines, evaluación estadística y generación de figuras corren íntegramente en CPU.

---

## 3. Dataset, Espacio de Características y Alineación de Calendarios

El dataset abarca desde el **4 de enero de 2016** hasta el **30 de octubre de 2026** ($3.953$ observaciones diarias), particionado de forma estrictamente causal en dos fases mediante `PHASE_A_END_DATE = "2025-12-24"` en `src/data_loader.py`:

* **Fase A (Selección y Desarrollo):** `2016-01-04` a `2025-12-24` ($3.643$ observaciones totales; $1.095$ días iniciales de ajuste + $14$ pliegues fuera de muestra = **$2.548$ días OOS**). Es la única partición visible para EDA, grilla de `EGARCH-X` y *sweeps* bayesianos.
* **Fase B (Test Ciego 2026):** `2025-12-25` a `2026-10-30` ($2$ pliegues fuera de muestra = **$310$ días OOS**). Permanece congelada durante todo el diseño experimental.
* **Muestra Completa Out-of-Sample:** $16$ pliegues concatenados (**$T = 2.858$ días**, 2019–2026).

### Variable objetivo y 10 canales estacionarios

* **Target (`target_r2_t1`):** $y_t = r_{\text{BTC}, t+1}^2$, con $r_{\text{BTC}, t} = \ln(C_t / C_{t-1})$.
* **Canales endógenos de Bitcoin (4):** variaciones intradiarias ancladas a la apertura `BTC_r_oc` ($\ln(C_t/O_t)$), `BTC_r_ho` ($\ln(H_t/O_t)$), `BTC_r_lo` ($\ln(L_t/O_t)$) y variación logarítmica de volumen `BTC_r_vol` ($\ln(V_t/V_{t-1})$).
* **Canales exógenos del S&P 500 / SPY (5):** `SPY_r_oc`, `SPY_r_ho`, `SPY_r_lo`, `SPY_r_vol` y retorno fuera de horario `SPY_r_overnight` ($\ln(O_t/C_{t-1})$).
* **Canal exógeno de estrés macro (1):** `VIX_log` ($\ln(\text{VIX}_t)$).

### Tratamiento estructural de fines de semana y feriados

Para preservar el calendario continuo $24/7$ de Bitcoin ($2.723$ días hábiles NYSE y $1.230$ jornadas sin rueda) sin inyectar actividad bursátil ficticia, el pipeline distingue entre **variables de flujo** y **variables de estado**:

* **Canales de flujo de SPY (5 canales):** se calculan exclusivamente sobre los $2.723$ días hábiles *antes* del cruce temporal (`left join`) con Bitcoin y se imputan con **`0.0` estricto** en los $1.230$ días sin rueda bursátil. El salto acumulado durante el cierre se absorbe íntegramente en `SPY_r_overnight` al reabrir el mercado.
* **Variable de estado (`VIX_log`):** se propaga hacia adelante (`ffill`) desde el último cierre hábil disponible, manteniendo continuo el régimen de volatilidad implícita esperada.

---

## 4. Diseño Arquitectónico y Protocolo Experimental

### 4.1 Parametrización en Log-Varianza y Pérdida QLIKE

Las tres familias culminan en una proyección lineal sin restricción de signo (`nn.Linear(d, 1)`) inicializada con sesgo informativo $b_0 = -6.5$ ($\hat{\sigma}_0^2 \approx 1.50 \times 10^{-3}$), que predice directamente la log-varianza $\hat{\eta}_t \in \mathbb{R}$. La conversión a varianza estrictamente positiva se integra dentro de `QLIKELoss`:

$$\hat{\sigma}_{t+1}^2 = \max\!\left(\exp(\operatorname{clip}(\hat{\eta}_t, -20, 15)),\, 10^{-6}\right), \qquad \mathcal{L}_{\text{QLIKE}}(y_t, \hat{\sigma}_{t+1}^2) = \frac{y_t}{\hat{\sigma}_{t+1}^2} - \ln\!\left(\frac{y_t}{\hat{\sigma}_{t+1}^2}\right) - 1$$

Esta reparametrización vuelve al gradiente invariante a la escala ($\partial \mathcal{L} / \partial \hat{\eta}_t = 1 - y_t / \hat{\sigma}_{t+1}^2$) y se combina con acotamiento de norma de gradiente (`clip_grad_norm_` con umbral `1.0`). Tanto el target como todas las predicciones (econométricas y neuronales) se evalúan en escala decimal con piso común $\epsilon = 10^{-6}$.

### 4.2 Familias de Arquitecturas (`src/models.py`)

| Familia | Clase PyTorch | Mecanismo Temporal sobre Ventana Finita $W$ | Parámetros ($H=32, L=1$) |
| :--- | :--- | :--- | :---: |
| **CNN 1D** | `VolatilidadCNN` | Convolución 1D con kernels impares $K \in \{3, 5, 7\}$, *padding* simétrico, activación `GELU` y `AdaptiveAvgPool1d(2)` | `1.697` ($K=5$) |
| **GRU** | `VolatilidadGRU` | Red recurrente *stateless* con 2 compuertas ($z_\tau, r_\tau$) y memoria truncada al horizonte $W$ | `4.257` |
| **LSTM** | `VolatilidadLSTM` | Red recurrente *stateless* con celda aditiva y 3 compuertas ($f_\tau, i_\tau, o_\tau$) sobre ventana $W$ | `5.665` |

### 4.3 Walk-Forward Causal, Sweeps Bayesianos y Ensambles Jerárquicos

* **Ventana móvil fija de $1.095$ días (3 años) con paso semestral de $182$ días** ($128$ días en el Pliegue 15): dentro de cada pliegue, los primeros **$1.005$ días** ajustan el `RobustScaler` y entrenan pesos con `AdamW` (hasta 45 épocas), mientras los **$90$ días** posteriores actúan como validación interna aislada para *Early Stopping* (paciencia $= 10$, $\Delta = 10^{-4}$). Para no perder el arranque de cada bloque, se antepone un contexto causal (`features_context`) de los últimos $W-1$ días escalados del bloque previo.
* **Búsqueda bayesiana en Fase A:** 40 corridas por familia ($120$ totales) en Weights & Biases minimizando `avg_cross_val_loss` (`val_mean_qlike`) sobre `timesteps` ($W \in \{5, 7, 14, 21\}$), `hidden_size` ($H \in \{16, 24, 32\}$), `num_layers` ($L \in \{1, 2\}$), `kernel_size` ($K \in \{3, 5, 7\}$ para CNN), `dropout_rate` ($\{0.2, 0.3, 0.4, 0.5\}$), `batch_size` ($\{16, 32, 64\}$), `learning_rate` ($\text{Log-U}[3\times 10^{-4}, 3\times 10^{-3}]$) y `weight_decay` ($\text{Log-U}[10^{-5}, 10^{-2}]$).
* **Selección dual $2+2$ y experimento multisemilla:** tras exigir superar al `Benchmark_Promedio` en Fase A ($\text{QLIKE medio} \le 1.9623$), se congelan 4 configuraciones disjuntas por familia (`MeanTop1/2` + `MedTop1/2` = 12 configuraciones) y se reentrenan desde cero en los 16 pliegues sobre 5 semillas independientes (`[42, 123, 456, 789, 7122001]`), promediando predicciones en espacio de varianza para conformar ensambles de ablación (`Mean_Top2`, `Median_Top2`), híbridos por familia (`Hybrid_2x2`, 20 modelos) y el super-ensamble global (`DL_SuperEnsemble_60M`, 60 modelos).

---

## 5. Orden de Ejecución Reproducible

Ejecutar los scripts desde la raíz del repositorio en la siguiente secuencia:

```bash
# 1. Ingesta, alineación temporal y generación del dataset procesado
python scripts/1_data_builder.py

# 2. Análisis exploratorio y diagnóstico lineal/no lineal (estrictamente sobre Fase A)
python scripts/2_eda_regression.py

# 3. Selección de especificación EGARCH-X en Fase A y evaluación de los 3 baselines (16 folds)
python scripts/3_tune_egarch_x.py
python scripts/4_evaluate_baselines.py

# 4. Optimización bayesiana de hiperparámetros en Weights & Biases (solo Fase A)
wandb login
python scripts/5_run_sweep_cnn.py --count 40
python scripts/6_run_sweep_gru.py --count 40
python scripts/7_run_sweep_lstm.py --count 40

# 5. Selección dual 2+2, entrenamiento multisemilla (60 trayectorias) y ensambles
python scripts/8_run_experiments_seeds.py --select-only   # Inspección previa opcional
python scripts/8_run_experiments_seeds.py

# 6. Comparación estadística (Diebold-Mariano HLN + sensibilidad HAC Newey-West) y diagnósticos
python scripts/9_compare_dl_vs_baselines.py
python scripts/10_error_analysis_and_diagnostics.py

# 7. Generación de figuras individuales del informe (results/figures/report_individual/)
python scripts/11_prediction_plots.py
```

---

## 6. Síntesis de Resultados Principales

### Desempeño Out-of-Sample Consolidado (QLIKE)

| Modelo | Fase A ($T=2.548$) Mean / Median | Fase B Ciega 2026 ($T=310$) Mean / Median | Muestra Completa ($T=2.858$) Mean / Median | Win-Rate vs. `EGARCH_GED` (Completa) |
| :--- | :---: | :---: | :---: | :---: |
| `Benchmark_Promedio` | `1.9623` / `1.2758` | `1.8823` / `1.1332` | `1.9537` / `1.2587` | `31.67%` |
| `EGARCH(1,3)_GED` | `1.9025` / `1.1126` | `1.7948` / `0.9934` | `1.8908` / `1.1069` | `50.00%` |
| `EGARCH(2,3)-X_GED` | `1.8851` / `1.1160` | **`1.7613`** / `1.0809` | `1.8717` / `1.1129` | `46.22%` |
| `CNN_1D_Hybrid_2x2` | `1.8893` / `1.1765` | `1.7710` / `1.0516` | `1.8765` / `1.1646` | `42.93%` |
| `GRU_Hybrid_2x2` | `1.8390` / **`1.1064`** | `1.7787` / **`0.9702`** | `1.8325` / **`1.0841`** | `48.99%` |
| `LSTM_Hybrid_2x2` | **`1.8328`** / `1.1077` | `1.7886` / `0.9777` | **`1.8280`** / `1.0942` | **`50.07%`** |
| `DL_SuperEnsemble_60M` | `1.8436` / `1.1356` | `1.7648` / `1.0164` | `1.8351` / `1.1184` | `46.61%` |

### Hallazgos Estadísticos Clave (Test de Diebold-Mariano HLN 1997 + Sensibilidad HAC)

* **Rechazo de superioridad global del Deep Learning sobre `EGARCH(1,3)_GED`:** si bien todos los ensambles neuronales superan holgadamente al `Benchmark_Promedio` ($p_{\text{bilat.}} < 0.0001$), ninguna arquitectura supera estadísticamente al modelo econométrico univariado al 5% (`LSTM_Hybrid_2x2`: $p_{\text{bilat.}} = 0.1312$, $p_{\text{unilat.}} = 0.0656$; verificado sin cambios bajo HAC Newey-West con $L = \lfloor T^{1/3} \rfloor$: $p_{\text{bilat.}} = 0.1271$).
* **Paridad en el Test Ciego 2026:** en un régimen de menor turbulencia, `EGARCH(2,3)-X_GED` (`1.7613`) y `DL_SuperEnsemble_60M` (`1.7648`) empatan estadísticamente ($p_{\text{bilat.}} = 0.9465$), mientras `GRU_Hybrid_2x2` lidera en mediana (`0.9702`) y *win-rate* diario (`60.65%`).
* **Jerarquía interna y parsimonia:** dentro del aprendizaje profundo, las redes recurrentes superan significativamente a la CNN 1D (`GRU`: $p_{\text{bilat.}} = 0.0001$; `LSTM`: $p_{\text{bilat.}} = 0.0012$), mientras que GRU y LSTM empatan estadísticamente ($p_{\text{bilat.}} = 0.7066$) con 25% menos parámetros recurrentes en GRU.
* **Fortalezas complementarias por régimen:** los modelos EGARCH dominan en baja volatilidad ($< P_{50}$) y eliminan por completo la autocorrelación ARCH residual ($p > 0.96$ en Ljung-Box y ARCH-LM sobre $z^2$), mientras que las redes neuronales superan significativamente a la econometría en volatilidad moderada ($P_{50}\text{--}P_{90}$, $p < 0.0003$), logran una calibración de varianza residual más cercana a $1$ ($\text{Var}(z) = 1.0140$ en LSTM) y reducen un **32.7%** el QLIKE medio en los 15 mayores shocks históricos.