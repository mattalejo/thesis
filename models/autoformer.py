import torch
import torch.nn as nn
from transformers import AutoformerConfig, AutoformerModel


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(
        self,
        d_model: int = 64,
        horizon: int = 1,
        dropout: float = 0.2,
        activation: str = "gelu"
    ):
        super(Autoformer, self).__init__()
        self.model_type = "Autoformer"
        self.configuration = AutoformerConfig(
            d_model=d_model,
            prediction_length=horizon,
            dropout=dropout,
            activation_function=activation,
        )
        self.autoformer = AutoformerModel(self.configuration)

    def forward(self, src, tgt):
        src = src.squeeze(2)
        tgt = tgt.squeeze(2)
        output = self.autoformer(past_values=src, future_values=tgt)
        return output





