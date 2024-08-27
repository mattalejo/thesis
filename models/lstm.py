import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Long Short-Term Memory Network
    """
    def __init__(
        self, 
        input_size: int = 1,  # number of features
        hidden_size: int = 128, 
        output_size: int = 1,
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
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):  # src: (batch, seq_len, input_size)
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        out, (hn, cn) = self.lstm(src)
        print(f"0. Output shape after FC layer: {out.shape}")
        out = self.fc(out[:, -1, :])  # Get output from the last time step
        print(f"1. Output shape after FC layer: {out.shape}")
        return out
