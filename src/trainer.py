import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import wandb

from .data_loader import PHASE_A_END_DATE
from .evaluation import compute_qlike_vector

# Constantes numéricas canónicas compartidas entre entrenamiento y evaluación
MIN_LOG_VAR = -20.0
MAX_LOG_VAR = 15.0
DEFAULT_EPS = 1e-6


def set_deterministic_seed(seed: int = 42):
    """
    Garantiza reproducibilidad estricta en Python, NumPy, PyTorch y CuDNN.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_var_to_var_tensor(
    h_pred: torch.Tensor,
    eps: float = DEFAULT_EPS,
    min_h: float = MIN_LOG_VAR,
    max_h: float = MAX_LOG_VAR,
) -> torch.Tensor:
    """Convierte log-varianza h_pred en varianza estrictamente positiva hat_sigma^2."""
    h_clamped = torch.clamp(h_pred, min=min_h, max=max_h)
    return torch.clamp(torch.exp(h_clamped), min=eps)


def log_var_to_var_numpy(
    h_pred: np.ndarray,
    eps: float = DEFAULT_EPS,
    min_h: float = MIN_LOG_VAR,
    max_h: float = MAX_LOG_VAR,
) -> np.ndarray:
    """Versión NumPy idéntica a log_var_to_var_tensor para inferencia y métricas."""
    h_clamped = np.clip(np.asarray(h_pred, dtype=np.float64), min_h, max_h)
    return np.clip(np.exp(h_clamped), eps, None)


class QLIKELoss(nn.Module):
    """
    Función de pérdida asimétrica QLIKE para pronóstico de varianza condicional.
    Opera directamente sobre la salida en log-varianza h_pred = ln(hat_sigma^2).
    """
    def __init__(self, eps: float = DEFAULT_EPS, max_h: float = MAX_LOG_VAR, min_h: float = MIN_LOG_VAR):
        super().__init__()
        self.eps = eps
        self.max_h = max_h
        self.min_h = min_h

    def forward(self, h_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        hat_sigma2 = log_var_to_var_tensor(h_pred, eps=self.eps, min_h=self.min_h, max_h=self.max_h)
        y_safe = torch.clamp(y_true, min=self.eps)

        ratio = y_safe / hat_sigma2
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)


def create_sequences(features, target, timesteps, features_context=None):
    """
    Construye tensores deslizantes (Ventanas, Timesteps, Canales).
    Si se provee features_context (los últimos timesteps-1 días del bloque previo),
    se predicen las len(features) observaciones completas sin perder el arranque del bloque.
    """
    if features_context is not None and timesteps > 1:
        if len(features_context) != (timesteps - 1):
            raise ValueError(
                f"features_context debe tener longitud timesteps - 1 ({timesteps - 1}), "
                f"recibido: {len(features_context)}"
            )
        full_features = np.vstack([features_context, features])
        X = [full_features[i : i + timesteps] for i in range(len(features))]
        y = target[: len(features)]
        return (
            torch.tensor(np.asarray(X), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.float32),
        )
    else:
        X, y = [], []
        for i in range(len(features) - timesteps + 1):
            X.append(features[i : i + timesteps])
            y.append(target[i + timesteps - 1])
        return (
            torch.tensor(np.asarray(X), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.float32),
        )


def create_sequences_with_dates(features, target, dates, timesteps, features_context=None):
    """
    Construye tensores deslizantes vinculando cada ventana con la fecha exacta de inferencia t
    (cuyo target es r_{t+1}^2).
    """
    X_t, y_t = create_sequences(features, target, timesteps, features_context=features_context)
    if features_context is not None and timesteps > 1:
        d = np.asarray(dates[: len(features)])
    else:
        d = np.asarray([dates[i + timesteps - 1] for i in range(len(features) - timesteps + 1)])

    if len(X_t) != len(d):
        raise RuntimeError(f"Desalineación interna: X tiene {len(X_t)} muestras y dates tiene {len(d)}.")
    return X_t, y_t, d


def evaluate_model_qlike(model, data_loader, device, eps: float = DEFAULT_EPS) -> dict:
    """
    Evalúa determinísticamente (Dropout desactivado) las pérdidas QLIKE sobre un DataLoader.
    """
    model.eval()
    y_list, h_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            h_pred = model(X_batch).view(-1)
            h_list.extend(h_pred.cpu().numpy().tolist())
            y_list.extend(y_batch.cpu().numpy().tolist())

    y_arr = np.asarray(y_list, dtype=np.float64)
    h_arr = np.clip(np.asarray(h_list, dtype=np.float64), MIN_LOG_VAR, MAX_LOG_VAR)
    sigma2_arr = log_var_to_var_numpy(h_arr, eps=eps)
    qlike = compute_qlike_vector(y_arr, sigma2_arr, eps=eps)

    return {
        "mean_qlike": float(np.mean(qlike)),
        "median_qlike": float(np.median(qlike)),
        "std_daily": float(np.std(qlike, ddof=1)) if len(qlike) > 1 else 0.0,
        "qlike_arr": qlike,
        "y_true": y_arr,
        "log_var_pred": h_arr,
        "sigma2_pred": sigma2_arr,
    }


def train_fold(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    fold_idx,
    device,
    patience: int = 10,
    min_delta: float = 1e-4,
    max_grad_norm: float = 1.0,
    log_to_wandb: bool = False,
):
    best_val_loss = float("inf")
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            h_pred = model(X_batch).view(-1)
            loss = criterion(h_pred, y_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_n = X_batch.size(0)
            train_loss_sum += loss.item() * batch_n
            train_samples += batch_n

        train_loss_epoch = (train_loss_sum / train_samples) if train_samples > 0 else float("nan")

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                h_pred = model(X_batch).view(-1)
                loss = criterion(h_pred, y_batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_n = X_batch.size(0)
                    val_loss_sum += loss.item() * batch_n
                    val_samples += batch_n

        val_loss_epoch = (val_loss_sum / val_samples) if val_samples > 0 else float("nan")

        history["train_loss"].append(float(train_loss_epoch))
        history["val_loss"].append(float(val_loss_epoch))

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                f"fold_{fold_idx}_train_loss": train_loss_epoch,
                f"fold_{fold_idx}_val_loss": val_loss_epoch,
                f"fold_{fold_idx}_epoch": epoch,
            })

        if not np.isnan(val_loss_epoch) and val_loss_epoch < (best_val_loss - min_delta):
            best_val_loss = float(val_loss_epoch)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return best_val_loss, history


def _extract_config_value(config, key, default=None):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict) and key in config:
        return config[key]
    return default


def _execute_walk_forward_core(
    model_class,
    X_raw,
    y_raw,
    dates,
    config,
    train_window_days: int = 1095,
    internal_val_days: int = 90,
    val_window_days: int = 182,
    step_days: int = 182,
    epochs: int = 45,
    patience: int = 10,
    seed: int = 42,
    log_to_wandb: bool = False,
):
    """
    Núcleo unificado de evaluación Walk-Forward (Fase A: Folds 0-13 + Fase B: Folds 14-15):
    - Dentro de cada ventana de entrenamiento de 1,095 días:
        * Primeros (1,095 - 90) = 1,005 días: ajuste de RobustScaler y optimización de pesos.
        * Últimos 90 días: validación interna aislada para EarlyStopping.
    - Bloque de evaluación out-of-sample (182 días, o 127 días en Fold 15):
        * 100% ciego tanto al escalado como al criterio de parada temprana.
    """
    if internal_val_days <= 0 or internal_val_days >= train_window_days:
        raise ValueError("internal_val_days debe estar entre 1 y train_window_days - 1.")

    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"

    total_days = len(X_raw)
    input_size = X_raw.shape[1]

    if dates is None:
        dates_arr = np.arange(total_days)
    else:
        dates_arr = pd.to_datetime(dates).values
        if len(dates_arr) != total_days:
            raise ValueError("La longitud de 'dates' no coincide con 'X_raw'.")

    timesteps = int(_extract_config_value(config, "timesteps", 7))
    batch_size = int(_extract_config_value(config, "batch_size", 32))
    hidden_size = int(_extract_config_value(config, "hidden_size", 32))
    dropout_rate = float(_extract_config_value(config, "dropout_rate", 0.3))
    learning_rate = float(_extract_config_value(config, "learning_rate", 1e-3))
    weight_decay = float(_extract_config_value(config, "weight_decay", 1e-4))
    kernel_size = int(_extract_config_value(config, "kernel_size", 3))
    num_layers = int(_extract_config_value(config, "num_layers", 1))

    if (train_window_days - internal_val_days) <= timesteps:
        raise ValueError("La ventana efectiva de entrenamiento debe ser mayor que timesteps.")

    fold_metrics = []
    all_predictions = []
    all_histories = {}

    phase_a_cutoff = pd.Timestamp(PHASE_A_END_DATE)
    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_days:
        train_start = start_idx
        train_end = start_idx + train_window_days
        fit_end = train_end - internal_val_days
        test_start = train_end
        test_end = min(test_start + val_window_days, total_days)

        if (test_end - test_start) < 90:
            break

        # 1. Partición estrictamente causal: Fit (1005d) -> Internal Val (90d) -> OOS Test (182d)
        X_fit_raw = X_raw[train_start:fit_end]
        y_fit_raw = y_raw[train_start:fit_end]
        dates_fit = dates_arr[train_start:fit_end]

        X_int_raw = X_raw[fit_end:train_end]
        y_int_raw = y_raw[fit_end:train_end]
        dates_int = dates_arr[fit_end:train_end]

        X_test_raw = X_raw[test_start:test_end]
        y_test_raw = y_raw[test_start:test_end]
        dates_test = dates_arr[test_start:test_end]

        # 2. Escalado ajustado SOLO sobre los 1,005 días de fit
        scaler = RobustScaler()
        X_fit_scaled = scaler.fit_transform(X_fit_raw)
        X_int_scaled = scaler.transform(X_int_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. Contextos históricos causales para no perder días en validación interna ni en test
        ctx_for_int = X_fit_scaled[-(timesteps - 1) :] if timesteps > 1 else None
        ctx_for_test = X_int_scaled[-(timesteps - 1) :] if timesteps > 1 else None

        X_fit_seq, y_fit_seq, _ = create_sequences_with_dates(
            X_fit_scaled, y_fit_raw, dates_fit, timesteps
        )
        X_int_seq, y_int_seq, _ = create_sequences_with_dates(
            X_int_scaled, y_int_raw, dates_int, timesteps, features_context=ctx_for_int
        )
        X_test_seq, y_test_seq, d_test_seq = create_sequences_with_dates(
            X_test_scaled, y_test_raw, dates_test, timesteps, features_context=ctx_for_test
        )

        g = torch.Generator()
        g.manual_seed(seed + fold_idx)

        fit_loader = DataLoader(
            TensorDataset(X_fit_seq, y_fit_seq),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            pin_memory=use_pin_memory,
        )
        fit_eval_loader = DataLoader(
            TensorDataset(X_fit_seq, y_fit_seq),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        )
        int_val_loader = DataLoader(
            TensorDataset(X_int_seq, y_int_seq),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_seq, y_test_seq),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_pin_memory,
        )

        model_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "dropout_rate": dropout_rate,
            "num_layers": num_layers,
        }
        if model_class.__name__ == "VolatilidadCNN":
            model_kwargs["kernel_size"] = kernel_size

        model = model_class(**model_kwargs).to(device)
        criterion = QLIKELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # EarlyStopping guiado por int_val_loader (90d), nunca por test_loader
        best_int_val_loss, history = train_fold(
            model=model,
            train_loader=fit_loader,
            val_loader=int_val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            fold_idx=fold_idx,
            device=device,
            patience=patience,
            log_to_wandb=log_to_wandb,
        )

        # Evaluación final sobre los tres subconjuntos
        fit_eval = evaluate_model_qlike(model, fit_eval_loader, device)
        int_eval = evaluate_model_qlike(model, int_val_loader, device)
        test_eval = evaluate_model_qlike(model, test_loader, device)

        if dates is not None:
            ts_start = pd.Timestamp(d_test_seq[0])
            ts_end = pd.Timestamp(d_test_seq[-1])
            val_start_str = ts_start.strftime("%Y-%m-%d")
            val_end_str = ts_end.strftime("%Y-%m-%d")
            phase_label = "Fase_A_Seleccion" if ts_end <= phase_a_cutoff else "Fase_B_Test_Ciego"
        else:
            val_start_str = str(d_test_seq[0])
            val_end_str = str(d_test_seq[-1])
            phase_label = "Fase_A_Seleccion" if fold_idx <= 13 else "Fase_B_Test_Ciego"

        records_df = pd.DataFrame({
            "Date": pd.to_datetime(d_test_seq) if dates is not None else d_test_seq,
            "phase": phase_label,
            "fold": fold_idx,
            "seed": seed,
            "y_true": test_eval["y_true"],
            "log_var_pred": test_eval["log_var_pred"],
            "sigma2_pred": test_eval["sigma2_pred"],
            "qlike": test_eval["qlike_arr"],
        })
        all_predictions.append(records_df)

        fold_metrics.append({
            "phase": phase_label,
            "fold": fold_idx,
            "val_start": val_start_str,
            "val_end": val_end_str,
            "samples": len(test_eval["qlike_arr"]),
            "mean_qlike": test_eval["mean_qlike"],
            "val_mean_qlike": test_eval["mean_qlike"],
            "val_median_qlike": test_eval["median_qlike"],
            "val_std_daily": test_eval["std_daily"],
            "internal_val_mean_qlike": int_eval["mean_qlike"],
            "train_mean_qlike": fit_eval["mean_qlike"],
            "train_median_qlike": fit_eval["median_qlike"],
            "best_val_loss": best_int_val_loss,
        })
        all_histories[f"fold_{fold_idx}"] = history

        start_idx += step_days
        fold_idx += 1

    df_preds = pd.concat(all_predictions, ignore_index=True)
    df_metrics = pd.DataFrame(fold_metrics)
    return df_metrics, df_preds, all_histories


def run_walk_forward_sweep(
    model_class,
    X_raw,
    y_raw,
    config,
    epochs: int = 45,
    patience: int = 10,
    seed: int = 42,
    train_window_days: int = 1095,
    internal_val_days: int = 90,
    val_window_days: int = 182,
    step_days: int = 182,
):
    """
    Ejecuta el Walk-Forward con logging por época en W&B y reporta tanto mean como median QLIKE
    para la selección híbrida 2+2 al cerrar Fase A.
    """
    df_metrics, df_preds, _ = _execute_walk_forward_core(
        model_class=model_class,
        X_raw=X_raw,
        y_raw=y_raw,
        dates=None,
        config=config,
        train_window_days=train_window_days,
        internal_val_days=internal_val_days,
        val_window_days=val_window_days,
        step_days=step_days,
        epochs=epochs,
        patience=patience,
        seed=seed,
        log_to_wandb=True,
    )

    val_mean_qlike = float(df_metrics["val_mean_qlike"].mean())
    val_median_macro = float(df_metrics["val_median_qlike"].mean())
    val_median_micro = float(df_preds["qlike"].median())
    val_std_folds = float(df_metrics["val_mean_qlike"].std(ddof=1)) if len(df_metrics) > 1 else 0.0

    train_mean_qlike = float(df_metrics["train_mean_qlike"].mean())
    train_median_qlike = float(df_metrics["train_median_qlike"].mean())
    train_std_folds = float(df_metrics["train_mean_qlike"].std(ddof=1)) if len(df_metrics) > 1 else 0.0

    generalization_gap = float(val_mean_qlike - train_mean_qlike)

    summary_payload = {
        "val_mean_qlike": val_mean_qlike,
        "val_median_qlike": val_median_micro,
        "val_median_macro_qlike": val_median_macro,
        "val_std_folds": val_std_folds,
        "train_mean_qlike": train_mean_qlike,
        "train_median_qlike": train_median_qlike,
        "train_std_folds": train_std_folds,
        "generalization_gap": generalization_gap,
        "avg_cross_val_loss": val_mean_qlike,
    }

    if wandb.run is not None:
        wandb.log(summary_payload)
        for k, v in summary_payload.items():
            wandb.run.summary[k] = v

    return summary_payload


def run_walk_forward(
    model_class,
    X_raw,
    y_raw,
    dates,
    config,
    train_window_days: int = 1095,
    internal_val_days: int = 90,
    val_window_days: int = 182,
    step_days: int = 182,
    epochs: int = 45,
    patience: int = 10,
    seed: int = 42,
):
    """
    Ejecuta el Walk-Forward completo devolviendo métricas por fold, predicciones diarias
    vinculadas a fechas reales e historiales de entrenamiento por época.
    """
    return _execute_walk_forward_core(
        model_class=model_class,
        X_raw=X_raw,
        y_raw=y_raw,
        dates=dates,
        config=config,
        train_window_days=train_window_days,
        internal_val_days=internal_val_days,
        val_window_days=val_window_days,
        step_days=step_days,
        epochs=epochs,
        patience=patience,
        seed=seed,
        log_to_wandb=False,
    )