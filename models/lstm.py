import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM
    """
    def __init__(
        self, 
        input_size: int = 1,  # number of features
        hidden_size: int = 128, 
        horizon: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        super(LSTM, self).__init__()
        self.model_type = "LSTM"
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            dropout=dropout,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):  # x: (batch, seq_len, input_size)
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        out, _ = self.lstm(src)  #, h0) 
        out = self.fc(out[:, -1:])  # , :])  # Get output from the last time step
        return out
