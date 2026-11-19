import torch
import torch.nn as nn


def _validate_base_params(input_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
    if input_size <= 0 or hidden_size <= 0:
        raise ValueError(f"input_size y hidden_size deben ser > 0. Recibido: {input_size}, {hidden_size}")
    if num_layers not in (1, 2):
        raise ValueError(f"num_layers debe ser 1 o 2. Recibido: {num_layers}")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate debe estar en [0, 1). Recibido: {dropout_rate}")


class VolatilidadGRU(nn.Module):
    """
    Red recurrente GRU que predice la log-varianza condicional h_t = ln(hat_sigma^2_{t+1}).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.3,
        num_layers: int = 1,
        init_bias: float = -6.5,
        **kwargs,
    ):
        super().__init__()
        _validate_base_params(input_size, hidden_size, num_layers, dropout_rate)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :]
        dropped_out = self.dropout(last_step_out)
        return self.fc(dropped_out)


class VolatilidadLSTM(nn.Module):
    """
    Red recurrente LSTM que predice la log-varianza condicional h_t = ln(hat_sigma^2_{t+1}).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.3,
        num_layers: int = 1,
        init_bias: float = -6.5,
        **kwargs,
    ):
        super().__init__()
        _validate_base_params(input_size, hidden_size, num_layers, dropout_rate)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        dropped_out = self.dropout(last_step_out)
        return self.fc(dropped_out)


class VolatilidadCNN(nn.Module):
    """
    Red convolucional 1D sobre la ventana histórica [t-T+1, ..., t] que predice
    la log-varianza condicional h_t = ln(hat_sigma^2_{t+1}).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.3,
        kernel_size: int = 3,
        num_layers: int = 1,
        init_bias: float = -6.5,
        **kwargs,
    ):
        super().__init__()
        _validate_base_params(input_size, hidden_size, num_layers, dropout_rate)

        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(
                f"kernel_size debe ser un entero positivo impar para padding simétrico. Recibido: {kernel_size}"
            )

        self.num_layers = num_layers
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act1 = nn.GELU()

        if self.num_layers == 2:
            self.conv2 = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.act2 = nn.GELU()

        # Divide la ventana histórica en dos mitades (memoria lejana vs. reciente)
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, 1)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Timesteps, Channels) -> (Batch, Channels, Timesteps)
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(x))

        if self.num_layers == 2:
            x = self.act2(self.conv2(x))

        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.fc(x)