import torch
import torch.nn as nn

class VolatilidadGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(VolatilidadGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :]
        dropped_out = self.dropout(last_step_out)
        return self.fc(dropped_out)


class VolatilidadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(VolatilidadLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        dropped_out = self.dropout(last_step_out)
        return self.fc(dropped_out)


class VolatilidadCNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3, kernel_size=3, num_layers=1):
        super(VolatilidadCNN, self).__init__()
        self.num_layers = num_layers
        
        # padding='same' mantiene la longitud de la secuencia temporal constante
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding='same'
        )
        self.act1 = nn.GELU()
        
        # Segunda capa opcional controlada por el optimizador bayesiano
        if self.num_layers == 2:
            self.conv2 = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding='same'
            )
            self.act2 = nn.GELU()
            
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
        # Inicialización del bias para estabilizar el QLIKE inicial
        nn.init.constant_(self.fc.bias, -6.5)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.conv1(x))
        
        if self.num_layers == 2:
            x = self.act2(self.conv2(x))
            
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.fc(x)