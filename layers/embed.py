import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x, x_time):
        return x + self.pe[:x_time.size(0), :]


class Time2Vec(nn.Module):
    def __init__(self, feature_size):
        super(Time2Vec, self).__init__()
        self.feature_size = feature_size
        self.weights = nn.Parameter(torch.Tensor(self.feature_size))
        self.bias = nn.Parameter(torch.Tensor(self.feature_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, x_time):
        time_linear = self.weights * x_time + self.bias
        time_linear[:, 1:2] = torch.sin(time_linear[:, 1:2])
        print(time_linear,time_linear.shape)

        return x + time_linear