import torch
import torch.nn as nn

# prompt: create a CNN for time series forecasting with dimensionst (batch, seq_len, num_features). layers sequentially include conv1d, maxpool1d, conv1d, maxpool1d, dropout, flatten, linear, dropout, linear, relu, linear. output should be  (batch, horizon, num_features). implement in pytorch

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, seq_len: int = 128, filters: int = 64, num_features: int = 1, horizon: int = 1, dropout: float = 0.1):
        super(CNN, self).__init__()
        self.model_type = "CNN"
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=filters, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=60 * ((seq_len) // 4), out_features=128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=128, out_features=horizon * num_features)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=horizon * num_features, out_features=horizon * num_features)

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        x = src.permute(0, 2, 1)  # Reshape to (batch, num_features, seq_len)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.unsqueeze(1)
        # x = x.view(-1, horizon, num_features)  # Reshape to (batch, horizon, num_features)
        return x

