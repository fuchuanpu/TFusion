import torch.nn as nn
import torch
import math


class TimePositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super().__init__()

        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len * 2, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len * 2).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        _x = torch.zeros([x.size(0), x.size(1), self.d_model])
        for i in range(x.size(0)):
            _x[i] = torch.index_select(self.pe, 0, x[i])
        return _x.to(x.device)
