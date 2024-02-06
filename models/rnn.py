import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Recurrent Neural Network
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
        super(RNN, self).__init__()
        self.model_type = "RNN"
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            dropout=dropout,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt=None):  # x: (batch, seq_len, input_size)
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(src)  #, h0) 
        out = self.fc(out[:, -1:])  # , :])  # Get output from the last time step
        return out

