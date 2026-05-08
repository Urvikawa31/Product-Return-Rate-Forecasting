import torch
import torch.nn as nn

class QuantileRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2, model_type='GRU'):
        super(QuantileRNN, self).__init__()
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, _ = self.rnn(x)
        # We take the last time step
        out = self.fc(out[:, -1, :])
        return out

def get_rnn(input_dim, output_dim): return QuantileRNN(input_dim, output_dim, model_type='RNN')
def get_lstm(input_dim, output_dim): return QuantileRNN(input_dim, output_dim, model_type='LSTM')
def get_gru(input_dim, output_dim): return QuantileRNN(input_dim, output_dim, model_type='GRU')
