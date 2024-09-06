import torch.nn as nn
import torch

from .transformer import TransformerSelfAttenBlock
from .embedding import MultiModalEmbedding
from .traffusion import TransformerTrafficFusion

class TFusion(nn.Module):

    def __init__(self, 
                 flow_feature_size=6, 
                 interact_feature_size=12, 
                 hidden=100, 
                 n_layers=1, 
                 attn_heads=1, 
                 dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        # embedding for traffic, sum of positional, segment, token embeddings
        self.embedding = MultiModalEmbedding(embed_size=hidden)

        # multi-layers transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerSelfAttenBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )

        self.trans_soft = nn.Softmax(dim=-1)

        self.fus = TransformerTrafficFusion(hidden, hidden, flow_feature_size, interact_feature_size)

    def forward(self, x):
        x_seq, x_non_seq, x_interact = x
        # embedding the indexed sequence to sequence of vectors
        _x_seq = self.embedding(x_seq)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            _x_seq = transformer.forward(_x_seq)
        _x_seq = self.trans_soft(_x_seq)

        _x_fusion = self.fus(_x_seq[:, 0], x_non_seq, x_interact)

        return _x_fusion
