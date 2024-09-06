import torch
import torch.nn as nn

class TransformerTrafficFusion(nn.Module):
    def __init__(self, hidden, d_seq, d_non_seq, d_interact, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.seq_encoder = nn.Linear(d_seq, hidden)
        self.non_seq_encoder = nn.Linear(d_non_seq, hidden)
        self.interact_encoder = nn.Linear(d_interact, hidden)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_seq, x_non_seq, x_interact):
        Q_mat = self.interact_encoder(x_interact)
        K_mat = self.non_seq_encoder(x_non_seq)
        V_mat = self.seq_encoder(x_seq)

        # score = self.softmax(torch.matmul(Q_mat, K_mat.transpose(-2, -1))) / 10
        
        score = self.softmax(Q_mat * K_mat / 10)

        return  Q_mat + score * V_mat
