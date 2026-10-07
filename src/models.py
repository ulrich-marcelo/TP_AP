import torch
import torch.nn as nn

class VolatilidadGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super(VolatilidadGRU, self).__init__()
        
        # Implementación de la GRU. Al tener pocos datos, forzamos 1 sola capa (num_layers=1)
        # para restringir la capacidad y evitar overfitting.
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True # Fundamental: espera tensores (batch, seq_len, features)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Proyección final: Emite el logaritmo de la varianza (sin activación)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        out, h_n = self.gru(x)
        
        # out shape: (batch_size, timesteps, hidden_size)
        # Nos interesa exclusivamente la predicción en el instante T (el último de la ventana)
        last_step_out = out[:, -1, :]
        
        # Regularización explícita
        dropped_out = self.dropout(last_step_out)
        
        # Predicción del logaritmo de la varianza condicional
        return self.fc(dropped_out)