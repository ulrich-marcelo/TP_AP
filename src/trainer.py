import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import wandb

class QLIKELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, h_pred, y_true):
        hat_sigma2 = torch.exp(h_pred)
        hat_sigma2 = torch.clamp(hat_sigma2, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        ratio = y_true / hat_sigma2
        loss = ratio - torch.log(ratio) - 1.0
        return torch.mean(loss)

def create_sequences(features, target, timesteps):
    X, y = [], []
    for i in range(len(features) - timesteps):
        X.append(features[i:(i + timesteps)])
        y.append(target[i + timesteps - 1])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

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
            # Transferencia de tensores a GPU/CPU
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            h_pred = model(X_batch).squeeze()
            loss = criterion(h_pred, y_batch)
            loss.backward()

            # Gradient clipping estricto
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

                h_pred = model(X_batch).squeeze()
                loss = criterion(h_pred, y_batch)
                val_loss_epoch += loss.item() * X_batch.size(0)

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)

        wandb.log({
            f"fold_{fold_idx}_train_loss": train_loss_epoch,
            f"fold_{fold_idx}_val_loss": val_loss_epoch,
            "epoch": epoch
        })

        # Early Stopping por fold
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

def run_walk_forward_sweep(model_class, X_raw, y_raw, config, epochs=45, patience=10):
    """
    model_class: Clase de PyTorch sin instanciar (ej: VolatilidadGRU)
    X_raw, y_raw: Dataset completo
    config: Objeto de configuración de Weights & Biases
    """
    # Detección automática del dispositivo de cómputo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = (device.type == "cuda")

    total_days = len(X_raw)
    train_window = 3 * 365 
    val_window = 365 
    input_size = X_raw.shape[1]

    fold_val_losses = []
    start_idx = 0
    fold_idx = 0

    while start_idx + train_window < total_days:
        train_start = start_idx
        train_end = start_idx + train_window
        val_start = train_end
        val_end = min(val_start + val_window, total_days)

        if val_end - val_start < 180:
            break

        # 1. Particionado estricto
        X_train_fold = X_raw[train_start:train_end]
        y_train_fold = y_raw[train_start:train_end]
        X_val_fold = X_raw[val_start:val_end]
        y_val_fold = y_raw[val_start:val_end]

        # 2. RobustScaler aislado por fold
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # 3. Creación de secuencias
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_fold, config.timesteps)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_fold, config.timesteps)

        # 4. DataLoaders con memoria paginada para acelerar copia a GPU
        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        val_dataset = TensorDataset(X_val_seq, y_val_seq)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=use_pin_memory
        )

        # 5. Instanciación y migración del modelo al dispositivo
        model = model_class(
            input_size=input_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.dropout_rate
        ).to(device)

        criterion = QLIKELoss().to(device)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )

        # 6. Entrenamiento del fold
        best_fold_loss, _ = train_fold(
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

        fold_val_losses.append(best_fold_loss)

        start_idx += val_window
        fold_idx += 1

    avg_val_loss = float(np.mean(fold_val_losses))
    wandb.log({
        "avg_cross_val_loss": avg_val_loss,
        "device": str(device)
    })
    return avg_val_loss