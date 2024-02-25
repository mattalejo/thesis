import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
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
    def __init__(self, d_model, seq_len=128):
        super(Time2Vec, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.w0 = nn.Parameter(torch.randn(1, 1, 1)) 
        self.b0 = nn.Parameter(torch.randn(1, 1, 1))
        self.w = nn.Parameter(torch.Tensor(1, 1, self.d_model-1))
        self.b = nn.Parameter(torch.Tensor(1, 1, self.d_model-1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x, x_time):
        t2v = torch.zeros(self.seq_len, self.d_model)
        t2v[:, 0] = x_time * self.w0 + self.b0
        t2v[:, 1::] = torch.sin(x_time * self.w + self.b0)
        t2v = t2v.unsqueeze(0).transpose(0, 1)
        return x + t2v
