import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

class QLIKELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, h_pred, y_true):
        # h_pred es ln(sigma^2), la red predice la varianza en escala logarítmica
        hat_sigma2 = torch.exp(h_pred)
        hat_sigma2 = torch.clamp(hat_sigma2, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        ratio = y_true / hat_sigma2
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)

def create_sequences_with_dates(features, target, dates, timesteps):
    X, y, d = [], [], []
    for i in range(len(features) - timesteps):
        X.append(features[i:(i + timesteps)])
        y.append(target[i + timesteps - 1])
        d.append(dates[i + timesteps - 1])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
        np.array(d)
    )

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
    max_grad_norm=1.0
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

            optimizer.zero_grad()
            h_pred = model(X_batch).squeeze(-1)
            loss = criterion(h_pred, y_batch)
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
                h_pred = model(X_batch).squeeze(-1)
                loss = criterion(h_pred, y_batch)
                val_loss_epoch += loss.item() * X_batch.size(0)

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)

        if val_loss_epoch < best_val_loss - min_delta:
            best_val_loss = val_loss_epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss, history

def evaluate_fold_predictions(model, val_loader, val_dates, criterion, device):
    """
    Computa y retorna las predicciones desescaladas y métricas individuales.
    """
    model.eval()
    y_trues, h_preds = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            h = model(X_batch).squeeze(-1)
            h_preds.extend(h.cpu().numpy().tolist())
            y_trues.extend(y_batch.numpy().tolist())

    y_trues = np.array(y_trues)
    h_preds = np.array(h_preds)
    sigma2_pred = np.exp(h_preds)
    
    # QLIKE por día individual
    eps = 1e-6
    sigma2_safe = np.clip(sigma2_pred, eps, None)
    y_safe = np.clip(y_trues, eps, None)
    ratio = y_safe / sigma2_safe
    pointwise_qlike = ratio - np.log(ratio) - 1.0

    records = pd.DataFrame({
        'Date': val_dates,
        'y_true': y_trues,
        'h_pred': h_preds,
        'sigma2_pred': sigma2_pred,
        'qlike': pointwise_qlike
    })
    return records

def run_walk_forward(
    model_class, 
    X_raw, 
    y_raw, 
    dates, 
    config, 
    train_window_days=1095,  # 3 años
    val_window_days=182,     # 6 meses (~1 semestre)
    step_days=182,
    epochs=45, 
    patience=10,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_days = len(X_raw)
    input_size = X_raw.shape[1]

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

        if (val_end - val_start) < 90:  # Descarta fragmentos menores a un trimestre
            break

        # 1. Partición cronológica estricta
        X_train_fold = X_raw[train_start:train_end]
        y_train_fold = y_raw[train_start:train_end]
        dates_train = dates[train_start:train_end]

        X_val_fold = X_raw[val_start:val_end]
        y_val_fold = y_raw[val_start:val_end]
        dates_val = dates[val_start:val_end]

        # 2. RobustScaler ajustado estrictamente sobre el fold de entrenamiento
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # 3. Construcción de tensores con timesteps
        X_train_seq, y_train_seq, _ = create_sequences_with_dates(
            X_train_scaled, y_train_fold, dates_train, config['timesteps']
        )
        X_val_seq, y_val_seq, d_val_seq = create_sequences_with_dates(
            X_val_scaled, y_val_fold, dates_val, config['timesteps']
        )

        train_loader = DataLoader(
            TensorDataset(X_train_seq, y_train_seq), 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_seq, y_val_seq), 
            batch_size=config['batch_size'], 
            shuffle=False
        )

        # 4. Instanciación con kwargs completos (evita omisión de kernel y layers en CNN)
        model_kwargs = {
            'input_size': input_size,
            'hidden_size': config['hidden_size'],
            'dropout_rate': config['dropout_rate'],
            'num_layers': config.get('num_layers', 1)
        }
        if 'kernel_size' in config:
            model_kwargs['kernel_size'] = config['kernel_size']

        model = model_class(**model_kwargs).to(device)
        criterion = QLIKELoss().to(device)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )

        # 5. Entrenamiento
        best_loss, history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            fold_idx=fold_idx,
            device=device,
            patience=patience
        )

        # 6. Extracción de predicciones y métricas
        pred_df = evaluate_fold_predictions(model, val_loader, d_val_seq, criterion, device)
        pred_df['fold'] = fold_idx
        pred_df['seed'] = seed
        all_predictions.append(pred_df)

        fold_mean_qlike = float(pred_df['qlike'].mean())
        fold_median_qlike = float(pred_df['qlike'].median())

        fold_metrics.append({
            'fold': fold_idx,
            'mean_qlike': fold_mean_qlike,
            'median_qlike': fold_median_qlike,
            'best_val_loss': best_loss
        })
        all_histories[f'fold_{fold_idx}'] = history

        start_idx += step_days
        fold_idx += 1

    df_preds = pd.concat(all_predictions, ignore_index=True)
    df_metrics = pd.DataFrame(fold_metrics)
    
    return df_metrics, df_preds, all_histories