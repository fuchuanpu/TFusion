import torch
import torch.nn as nn
from .time_position import TimePositionalEmbedding

class MultiModalEmbedding(nn.Module):


    def __init__(self, 
                 embed_size, 
                 info_index=0, 
                 pos_index=1, 
                 packet_feature_vocab=3 * 1500, 
                 max_len=100, dropout=0.1):

        super().__init__()

        self.info_index = info_index
        self.pos_index = pos_index

        self.embed_size = embed_size
        self.position_encode = TimePositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.embed_layer = nn.Embedding(packet_feature_vocab, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seq):
        pkt_attr_vocb = seq[:, :, self.info_index].squeeze().to(torch.long)
        pkt_time_vocb = seq[:, :, self.pos_index].squeeze().to(torch.int)
        x = self.embed_layer(pkt_attr_vocb) + self.position_encode(pkt_time_vocb)
        return self.dropout(x)
