# Modelado y Predicción de la Varianza Diaria de Bitcoin mediante Aprendizaje Profundo (2016-2026)



Este repositorio contiene el código fuente, canalizaciones de datos y configuración experimental del Trabajo Final de la materia **Aprendizaje Profundo**. El proyecto modela y predice la varianza condicional diaria de Bitcoin (BTC) para el período 2016-2026 utilizando redes neuronales profundas (GRU, LSTM y CNN 1D) frente a un modelo econométrico de referencia $EGARCH(1,3)\_GED$.

A diferencia de los enfoques convencionales centrados en predecir el precio nominal —afectados por falta de estacionariedad y dependencia de escala—, el objetivo es predecir la volatilidad empleando el **retorno logarítmico diario al cuadrado ($r_{BTC, t+1}^2$)** como proxy observable no sesgado.

---

## 1. Estructura del Proyecto

```text
├── data/
│   ├── raw/
│   │   └── raw_data.csv               # Cotizaciones históricas descargadas (BTC, SPY, VIX)
│   └── processed/
│       └── processed_data.csv         # Tensor de 10 canales estacionarios + variable objetivo
├── notebooks/ (o scripts/)
│   └── data_builder.ipynb             # Inspección exploratoria y gráficos de no estacionariedad
├── scripts/
│   ├── data_builder.py                # Pipeline ETL para descarga y cálculo de features
│   ├── run_sweep_gru.py               # Optimización bayesiana en W&B para arquitectura GRU
│   ├── run_sweep_lstm.py              # Optimización bayesiana en W&B para arquitectura LSTM
│   └── run_sweep_cnn.py               # Optimización bayesiana en W&B para arquitectura CNN 1D
├── src/
│   ├── __init__.py                    # Inicialización del módulo y exportación de clases
│   ├── data_loader.py                 # Carga de tensores y división X, y
│   ├── models.py                      # Definición de arquitecturas (VolatilidadGRU, LSTM, CNN)
│   └── trainer.py                     # Loss QLIKE, Walk-Forward, Early Stopping y soporte CUDA
├── requirements.txt                   # Dependencias del entorno virtual
└── README.md                          # Documentación del proyecto

```

---

## 2. Instalación y Configuración del Entorno

### 2.1 Clonar el repositorio y crear entorno virtual

```bash
git clone https://github.com/ulrich-marcelo/tp_ap.git
cd tp_ap

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # En Linux/macOS
# .\venv\Scripts\activate       # En Windows

```

### 2.2 Instalación de Dependencias

Para instalar las dependencias básicas:

```bash
pip install -r requirements.txt

```

### 2.3 Aceleración por Hardware (CUDA / GPU)

Para entrenar en placas de video NVIDIA, se debe instalar la rueda oficial de PyTorch compilada con soporte CUDA (en lugar de la versión pura de CPU).

1. Verificar que el controlador NVIDIA esté activo en la máquina:
```bash
nvidia-smi

```


2. Instalar la versión de PyTorch adecuada según la versión máxima de CUDA soportada por el driver:
* **Para CUDA 12.x:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121

```


* **Para CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118

```




3. Comprobar la detección del dispositivo:
```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

```



El pipeline de `src/trainer.py` detectará automáticamente `torch.cuda.is_available()`, migrando los tensores y modelos a la VRAM con `pin_memory=True` y `non_blocking=True`.

---

## 3. Pipeline de Datos y Espacio de Características

El espacio de características multivariado comprende 10 canales continuos rigurosamente estacionarios más la variable objetivo, abarcando desde 2016 hasta 2026:

* **Variable Objetivo (`target_r2_t1`)**:

$$r_{BTC,t} = \ln\left(\frac{C_t}{C_{t-1}}\right) \implies y_t = r_{BTC, t+1}^2$$



* **Canales Endógenos de Bitcoin (4 canales)**:


* `BTC_r_oc`: Retorno Open-to-Close $\ln(C_t / O_t)$

* `BTC_r_ho`: Amplitud Máxima $\ln(H_t / O_t)$

* `BTC_r_lo`: Amplitud Mínima $\ln(L_t / O_t)$

* `BTC_r_vol`: Variación del Volumen $\ln(V_t / V_{t-1})$



* **Canales Exógenos del S&P 500 / SPY (5 canales)**:


* `SPY_r_oc`, `SPY_r_ho`, `SPY_r_lo`, `SPY_r_vol`

* `SPY_r_overnight`: Retorno nocturno y fin de semana $\ln(O_t / C_{t-1})$



* **Canal Exógeno de Estrés Macro (1 canal)**:


* `VIX_log`: Transformación logarítmica del índice VIX $\ln(VIX_t)$




Para reconstruir el conjunto de datos crudo y procesado:

```bash
python scripts/data_builder.py

```

Los días no operativos de los mercados tradicionales (sábados, domingos y feriados) se alinean contra el calendario ininterrumpido 24/7/365 de Bitcoin mediante propagación del último valor conocido (`ffill`).

---

## 4. Metodología de Modelado y Entrenamiento

### 4.1 Función de Pérdida Asimétrica (QLIKE)

Durante la optimización se minimiza la función de pérdida cuasi-verosímil QLIKE:


$$QLIKE(y, \hat{\sigma}^2) = \frac{y}{\hat{\sigma}^2} - \ln\left(\frac{y}{\hat{\sigma}^2}\right) - 1$$


En concordancia con los modelos de la familia EGARCH:

* Las redes neuronales predicen directamente el logaritmo de la varianza condicional: $h_{pred} = \ln(\hat{\sigma}^2)$.


* La clase `QLIKELoss` en `src/trainer.py` transforma la predicción mediante $\hat{\sigma}^2 = \exp(h_{pred})$ y aplica un suelo numérico `torch.clamp(min=1e-6)` para garantizar positividad estricta sin inestabilidad de gradientes.



### 4.2 Validación Temporal: Walk-Forward (Ventana Deslizante)

Para impedir la filtración de información futura (*data leakage*), se utiliza una partición secuencial con ventana deslizante:

* **Ventana de entrenamiento fija**: 3 años (1095 días).


* **Ventana de validación *out-of-sample***: 1 año (365 días).


* **Estandarización aislada**: Se ajusta dinámicamente un `RobustScaler()` sobre el conjunto de entrenamiento de cada fold y se transforma tanto el entrenamiento como la ventana de validación correspondiente.


* El esquema genera 7 folds secuenciales cubriendo de 2016 a fines de 2025.



### 4.3 Regularización y Early Stopping

Dado el tamaño crítico de la serie (~3.500 observaciones):

* Se imponen celdas de capacidad reducida (16 a 32 unidades/filtros).


* Capas de Dropout agresivas (0.2 a 0.5).


* Optimizador `AdamW` con decaimiento de pesos (*weight decay*) desacoplado.


* Criterio de parada temprana (**Early Stopping** con paciencia de 10 épocas y umbral $\Delta = 10^{-4}$) para prevenir la memorización del ruido estocástico financiero.

---

## 5. Arquitecturas Disponibles

| Arquitectura | Clase en `src/models.py` | Tipo de Sistema Dinámico | Características

 |
| --- | --- | --- | --- |
| **GRU** | `VolatilidadGRU`<br> | Respuesta Infinita (2 compuertas)

 | Eficiencia paramétrica y menor varianza muestral.

 |
| **LSTM** | `VolatilidadLSTM` | Respuesta Infinita (3 compuertas)

 | Celda de estado aditiva para dependencias temporales largas.

 |
| **CNN 1D** | `VolatilidadCNN` | Respuesta Finita (Filtros locales)

 | Extracción convolucional de patrones de shock y MaxPool temporal.

 |

---

## 6. Ejecución de Sweeps con Weights & Biases (W&B)

Para garantizar un *benchmarking* imparcial, cada familia arquitectónica dispone de su propio script de optimización bayesiana con un presupuesto equivalente de 40 corridas (`runs`).

Antes de ejecutar, iniciar sesión en Weights & Biases:

```bash
wandb login

```

### Ejecución individual por arquitectura

* **Optimizar modelo GRU:**
```bash
python scripts/run_sweep_gru.py

```


* **Optimizar modelo LSTM:**
```bash
python scripts/run_sweep_lstm.py

```


* **Optimizar modelo CNN 1D:**
```bash
python scripts/run_sweep_cnn.py

```



### Espacio de Hiperparámetros Evaluado

* `learning_rate`: Muestreo logarítmico continuo entre $3 \times 10^{-4}$ y $3 \times 10^{-3}$.
* `weight_decay`: Muestreo logarítmico continuo entre $1 \times 10^{-5}$ y $1 \times 10^{-2}$.
* `hidden_size`: $\{16, 24, 32\}$ unidades / filtros.


* `dropout_rate`: $\{0.2, 0.3, 0.4, 0.5\}$.


* `timesteps`: $\{5, 7, 14, 21\}$ días históricos.
* `batch_size`: $\{16, 32, 64\}$.

---

## 7. Comparación de Resultados y Evaluación Estadística

El objetivo metodológico es contrastar las arquitecturas frente al modelo base de la econometría clásica obtenido bajo la misma metodología de validación:

* **Baseline Econométrico ($EGARCH(1,3)\_GED$):** $\text{QLIKE} = \mathbf{1.831}$.


* **Varianza Incondicional Constante (Benchmark Cero):** $\text{QLIKE} \approx \mathbf{2.074}$.
* **Mejor Modelo Deep Learning (GRU Óptima):** $\text{QLIKE} \approx \mathbf{1.770}$.

### Test de Significancia Estadística: Diebold-Mariano

Para corroborar que la ventaja de la red neuronal sobre el modelo econométrico es estadísticamente significativa y no producto del ruido estocástico del proxy, se ejecuta el test de Diebold-Mariano (1995) con corrección de Harvey, Leybourne y Newbold (1997) bajo la función de pérdida robusta QLIKE:

$$H_0: \mathbb{E}[QLIKE_{EGARCH, t} - QLIKE_{DL, t}] \le 0 \quad \text{vs.} \quad H_1: \mathbb{E}[QLIKE_{EGARCH, t} - QLIKE_{DL, t}] > 0$$

Un valor de $p < 0.05$ confirma la superioridad predictiva out-of-sample de la arquitectura seleccionada sobre el estándar econométrico clásico.