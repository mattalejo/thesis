import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Convolutional Neural Network for Time Series Forecasting
    """
    def __init__(
        self, 
        input_size: int = 128,  # number of past values
        output_size: int = 1,  # number of forecast days
        filters: int = 64,
        dropout: float = 0.5
    ):
        super(CNN, self).__init__()
        self.model_type = "CNN"
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters * (input_size // 4), 64)  # Adjust input size based on conv layers
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 64)
        self.activation = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, src, src_time=None, tgt=None, tgt_time=None):\
        print(src.shape)
        src_time, tgt, tgt_time = None, None, None  # Garbage collection
        x = self.conv1(src)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
