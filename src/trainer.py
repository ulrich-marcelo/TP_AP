import copy
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

def set_deterministic_seed(seed=42):
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

class QLIKELoss(nn.Module):
    """
    Función de pérdida asimétrica QLIKE para pronóstico de varianza condicional.
    Opera directamente sobre la salida en log-varianza h_pred = ln(hat_sigma^2).
    """
    def __init__(self, eps=1e-6, max_h=15.0, min_h=-20.0):
        super(QLIKELoss, self).__init__()
        self.eps = eps
        self.max_h = max_h
        self.min_h = min_h

    def forward(self, h_pred, y_true):
        # Clamping en espacio logarítmico para evitar desbordes de exp()
        h_clamped = torch.clamp(h_pred, min=self.min_h, max=self.max_h)
        hat_sigma2 = torch.exp(h_clamped)
        
        hat_sigma2 = torch.clamp(hat_sigma2, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        
        ratio = y_true / hat_sigma2
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)

def create_sequences(features, target, timesteps, features_context=None):
    """
    Construye tensores deslizantes (Ventanas, Timesteps, Canales).
    Si se provee features_context (ej. los últimos timesteps-1 días del fold previo),
    se predicen todas y cada una de las filas de features sin perder observaciones.
    """
    if features_context is not None and timesteps > 1:
        full_features = np.vstack([features_context, features])
        X = []
        for i in range(len(features)):
            X.append(full_features[i : i + timesteps])
        y = target[:len(features)]
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)
    else:
        X, y = [], []
        # Corrección del off-by-one (+ 1) para aprovechar la serie completa
        for i in range(len(features) - timesteps + 1):
            X.append(features[i : i + timesteps])
            y.append(target[i + timesteps - 1])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

def create_sequences_with_dates(features, target, dates, timesteps, features_context=None):
    """
    Construye tensores deslizantes vinculando las fechas reales de inferencia.
    """
    if features_context is not None and timesteps > 1:
        full_features = np.vstack([features_context, features])
        X = []
        for i in range(len(features)):
            X.append(full_features[i : i + timesteps])
        y = target[:len(features)]
        d = dates[:len(features)]
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
            np.array(d)
        )
    else:
        X, y, d = [], [], []
        for i in range(len(features) - timesteps + 1):
            X.append(features[i : i + timesteps])
            y.append(target[i + timesteps - 1])
            d.append(dates[i + timesteps - 1])
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
            np.array(d)
        )

def evaluate_model_qlike(model, data_loader, device, eps=1e-6):
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

    y_arr = np.array(y_list, dtype=np.float64)
    h_arr = np.clip(np.array(h_list, dtype=np.float64), -20.0, 15.0)
    sigma2_arr = np.exp(h_arr)

    sigma2_safe = np.clip(sigma2_arr, eps, None)
    y_safe = np.clip(y_arr, eps, None)
    ratio = y_safe / sigma2_safe
    qlike = ratio - np.log(ratio) - 1.0

    return {
        'mean_qlike': float(np.mean(qlike)),
        'median_qlike': float(np.median(qlike)),
        'std_daily': float(np.std(qlike, ddof=1)) if len(qlike) > 1 else 0.0,
        'qlike_arr': qlike,
        'y_true': y_arr,
        'sigma2_pred': sigma2_arr
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
    patience=10, 
    min_delta=1e-4, 
    max_grad_norm=1.0,
    log_to_wandb=False
):
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            h_pred = model(X_batch).view(-1)
            loss = criterion(h_pred, y_batch)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            train_loss_epoch += loss.item() * X_batch.size(0)

        train_loss_epoch /= len(train_loader.dataset)

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                h_pred = model(X_batch).view(-1)
                loss = criterion(h_pred, y_batch)
                val_loss_epoch += loss.item() * X_batch.size(0)

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                f"fold_{fold_idx}_train_loss": train_loss_epoch,
                f"fold_{fold_idx}_val_loss": val_loss_epoch,
                f"fold_{fold_idx}_epoch": epoch
            })

        if not np.isnan(val_loss_epoch) and val_loss_epoch < (best_val_loss - min_delta):
            best_val_loss = val_loss_epoch
            # Almacenar en CPU para liberar VRAM en GPU
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restaurar el mejor punto de control encontrado
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return best_val_loss, history

def _extract_config_value(config, key, default=None):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict) and key in config:
        return config[key]
    return default

def run_walk_forward_sweep(
    model_class, 
    X_raw, 
    y_raw, 
    config, 
    epochs=45, 
    patience=10,
    seed=42,
    train_window_days=1095,
    val_window_days=182,
    step_days=182
):
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_days = len(X_raw)
    input_size = X_raw.shape[1]

    timesteps = _extract_config_value(config, 'timesteps', 7)
    batch_size = _extract_config_value(config, 'batch_size', 32)
    hidden_size = _extract_config_value(config, 'hidden_size', 32)
    dropout_rate = _extract_config_value(config, 'dropout_rate', 0.3)
    learning_rate = _extract_config_value(config, 'learning_rate', 1e-3)
    weight_decay = _extract_config_value(config, 'weight_decay', 1e-4)
    kernel_size = _extract_config_value(config, 'kernel_size', 3)
    num_layers = _extract_config_value(config, 'num_layers', 1)

    fold_val_means = []
    fold_val_medians = []
    fold_train_means = []
    fold_train_medians = []

    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_days:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_days)

        if (val_end - val_start) < 90:
            break

        X_train_fold = X_raw[train_start:train_end]
        y_train_fold = y_raw[train_start:train_end]
        X_val_fold = X_raw[val_start:val_end]
        y_val_fold = y_raw[val_start:val_end]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Contexto histórico del final del set de entrenamiento para cubrir los 182 días
        X_train_context = X_train_scaled[-(timesteps - 1):] if timesteps > 1 else None

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_fold, timesteps)
        X_val_seq, y_val_seq = create_sequences(
            X_val_scaled, y_val_fold, timesteps, features_context=X_train_context
        )

        g = torch.Generator()
        g.manual_seed(seed + fold_idx)

        train_loader = DataLoader(
            TensorDataset(X_train_seq, y_train_seq), 
            batch_size=batch_size, 
            shuffle=True, 
            generator=g
        )
        train_eval_loader = DataLoader(
            TensorDataset(X_train_seq, y_train_seq), 
            batch_size=batch_size, 
            shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_seq, y_val_seq), 
            batch_size=batch_size, 
            shuffle=False
        )

        model_kwargs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
            'num_layers': num_layers
        }
        if model_class.__name__ == "VolatilidadCNN":
            model_kwargs['kernel_size'] = kernel_size

        model = model_class(**model_kwargs).to(device)
        criterion = QLIKELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            fold_idx=fold_idx,
            device=device,
            patience=patience,
            log_to_wandb=True
        )

        val_eval = evaluate_model_qlike(model, val_loader, device)
        train_eval = evaluate_model_qlike(model, train_eval_loader, device)

        fold_val_means.append(val_eval['mean_qlike'])
        fold_val_medians.append(val_eval['median_qlike'])
        fold_train_means.append(train_eval['mean_qlike'])
        fold_train_medians.append(train_eval['median_qlike'])

        start_idx += step_days
        fold_idx += 1

    val_mean_qlike = float(np.mean(fold_val_means))
    val_median_qlike = float(np.mean(fold_val_medians))
    val_std_folds = float(np.std(fold_val_means, ddof=1))

    train_mean_qlike = float(np.mean(fold_train_means))
    train_median_qlike = float(np.mean(fold_train_medians))
    train_std_folds = float(np.std(fold_train_means, ddof=1))

    generalization_gap = float(val_mean_qlike - train_mean_qlike)

    if wandb.run is not None:
        summary_payload = {
            "val_mean_qlike": val_mean_qlike,
            "val_median_qlike": val_median_qlike,
            "val_std_folds": val_std_folds,
            "train_mean_qlike": train_mean_qlike,
            "train_median_qlike": train_median_qlike,
            "train_std_folds": train_std_folds,
            "generalization_gap": generalization_gap,
            "avg_cross_val_loss": val_mean_qlike
        }
        wandb.log(summary_payload)
        for k, v in summary_payload.items():
            wandb.run.summary[k] = v

    return {
        'val_mean_qlike': val_mean_qlike,
        'val_median_qlike': val_median_qlike,
        'val_std_folds': val_std_folds,
        'train_mean_qlike': train_mean_qlike,
        'train_median_qlike': train_median_qlike,
        'train_std_folds': train_std_folds,
        'generalization_gap': generalization_gap
    }

def run_walk_forward(
    model_class, 
    X_raw, 
    y_raw, 
    dates, 
    config, 
    train_window_days=1095,
    val_window_days=182,
    step_days=182,
    epochs=45, 
    patience=10,
    seed=42
):
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_days = len(X_raw)
    input_size = X_raw.shape[1]

    timesteps = _extract_config_value(config, 'timesteps', 7)
    batch_size = _extract_config_value(config, 'batch_size', 32)
    hidden_size = _extract_config_value(config, 'hidden_size', 32)
    dropout_rate = _extract_config_value(config, 'dropout_rate', 0.3)
    learning_rate = _extract_config_value(config, 'learning_rate', 1e-3)
    weight_decay = _extract_config_value(config, 'weight_decay', 1e-4)
    kernel_size = _extract_config_value(config, 'kernel_size', 3)
    num_layers = _extract_config_value(config, 'num_layers', 1)

    fold_metrics = []
    all_predictions = []
    all_histories = {}

    start_idx = 0
    fold_idx = 0

    while start_idx + train_window_days < total_days:
        train_start = start_idx
        train_end = start_idx + train_window_days
        val_start = train_end
        val_end = min(val_start + val_window_days, total_days)

        if (val_end - val_start) < 90:
            break

        X_train_fold = X_raw[train_start:train_end]
        y_train_fold = y_raw[train_start:train_end]
        dates_train = dates[train_start:train_end]

        X_val_fold = X_raw[val_start:val_end]
        y_val_fold = y_raw[val_start:val_end]
        dates_val = dates[val_start:val_end]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Contexto histórico para no perder los primeros timesteps del semestre de prueba
        X_train_context = X_train_scaled[-(timesteps - 1):] if timesteps > 1 else None

        X_train_seq, y_train_seq, _ = create_sequences_with_dates(
            X_train_scaled, y_train_fold, dates_train, timesteps
        )
        X_val_seq, y_val_seq, d_val_seq = create_sequences_with_dates(
            X_val_scaled, y_val_fold, dates_val, timesteps, features_context=X_train_context
        )

        g = torch.Generator()
        g.manual_seed(seed + fold_idx)

        train_loader = DataLoader(
            TensorDataset(X_train_seq, y_train_seq), 
            batch_size=batch_size, 
            shuffle=True, 
            generator=g
        )
        train_eval_loader = DataLoader(
            TensorDataset(X_train_seq, y_train_seq), 
            batch_size=batch_size, 
            shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_seq, y_val_seq), 
            batch_size=batch_size, 
            shuffle=False
        )

        model_kwargs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
            'num_layers': num_layers
        }
        if model_class.__name__ == "VolatilidadCNN":
            model_kwargs['kernel_size'] = kernel_size

        model = model_class(**model_kwargs).to(device)
        criterion = QLIKELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_loss, history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            fold_idx=fold_idx,
            device=device,
            patience=patience,
            log_to_wandb=False
        )

        val_eval = evaluate_model_qlike(model, val_loader, device)
        train_eval = evaluate_model_qlike(model, train_eval_loader, device)

        records_df = pd.DataFrame({
            'Date': d_val_seq,
            'y_true': val_eval['y_true'],
            'sigma2_pred': val_eval['sigma2_pred'],
            'qlike': val_eval['qlike_arr'],
            'fold': fold_idx,
            'seed': seed
        })
        all_predictions.append(records_df)

        fold_metrics.append({
            'fold': fold_idx,
            'mean_qlike': val_eval['mean_qlike'],       # Compatibilidad directa
            'val_mean_qlike': val_eval['mean_qlike'],
            'val_median_qlike': val_eval['median_qlike'],
            'train_mean_qlike': train_eval['mean_qlike'],
            'train_median_qlike': train_eval['median_qlike'],
            'best_val_loss': best_loss,
            'samples': len(val_eval['qlike_arr'])
        })
        all_histories[f'fold_{fold_idx}'] = history

        start_idx += step_days
        fold_idx += 1

    df_preds = pd.concat(all_predictions, ignore_index=True)
    df_metrics = pd.DataFrame(fold_metrics)
    return df_metrics, df_preds, all_histories