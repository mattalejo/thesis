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
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):  # x: (batch, seq_len, input_size)
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        out, _ = self.rnn(src)  #, h0) 
        out = self.fc(out[:, -1:])  # , :])  # Get output from the last time step
        out = out.view(-1, self.horizon, 1)
        return out

class RNN5Day(nn.Module):
    """
    Recurrent Neural Network modified to output a sequence of a specified length (horizon).
    """
    def __init__(
        self, 
        input_size: int = 1,  # Number of features in the input
        hidden_size: int = 128,  # Number of features in the hidden state 
        horizon: int = 5,  # Number of time steps to predict into the future
        num_layers: int = 4,  # Number of recurrent layers
        dropout: float = 0.1,  # Dropout rate
        batch_first: bool = True  # If True, then the input and output tensors are provided as (batch, seq, feature)
    ):
        super(RNN5Day, self).__init__()
        self.model_type = "RNN 5-day"
        self.hidden_size = hidden_size
        self.horizon = horizon  # Add horizon attribute
        self.batch_first = batch_first
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=batch_first
        )
        # Update the linear layer to output the horizon length for each feature
        self.fc = nn.Linear(hidden_size, horizon * input_size) 

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):  # x: (batch, seq_len, input_size)
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        print(src.shape)
        out, _ = self.rnn(src)  # Process input through RNN
        print(f"0. Output shape after RNN layer: {out.shape}")
        out = self.fc(out[:, -1, :])  # Apply the linear layer to the last time step of each sequence
        print(f"1. Output shape after FC layer: {out.shape}")
        # Reshape output to match the desired output shape (batch_size, horizon, input_size)
        out = out.view(-1, self.horizon, 1)
        return out
