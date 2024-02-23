import torch
import torch.nn as nn
from layers.embed import PositionalEncoding, Time2Vec


class Transformer(nn.Module):
    """
    Transformer model
    """

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        nhead: int = 4,
        activation: str = "relu",
        batch_first: bool = True,
        embed_mode: str = "t2v"
    ):
        super(Transformer, self).__init__()
        self.model_type = "Transformer"
        # self.src_mask = None
        if embed_mode == "pe":
            self.pos_encoder = PositionalEncoding(d_model)
        elif embed_mode == "t2v":
            self.pos_encoder = Time2Vec(d_model)
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=encoder_head, dropout=dropout, activation=activation
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=num_layers
        # )
        # self.decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model, nhead=decoder_head, dropout=dropout, activation=activation
        # )
        # self.decoder = 
        self.decoder = nn.Linear(d_model, 1)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout, activation=activation, batch_first=batch_first)

    def forward(self, src, tgt, src_time, tgt_time):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     mask = self._generate_square_subsequent_mask(len(src))
        #     self.src_mask = mask
        # print(f"src shape before PE: {src.shape}")
        # src = self.pos_encoder(src)
        # print(f"src shape: {src.shape}")
        # output = self.transformer_encoder(src, self.src_mask)
        # output = self.decoder(output)
        # print(f"output shape: {output.shape}")

        src = self.pos_encoder(src, src_time)
        tgt = self.pos_encoder(src, tgt_time)
        output = self.transformer(src, tgt)
        output = self.decoder(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
