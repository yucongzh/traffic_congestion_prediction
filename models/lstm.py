import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.nn.functional as F
import torch.optim as optim


class LSTMNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, latent_dim, n_layer, bsz):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, latent_dim, n_layer)
        self.L1 = nn.Linear(latent_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, X, state1, bsz):
        x = x.transpose(1,2)
        x = x.reshape(-1, bsz, self.out_dim, 1)
        x = torch.mul(x, X)
        x = x.reshape(-1, bsz, self.in_dim)
        x, state1 = self.lstm(x, state1)
        extra = x
        x = self.L1(x)
        # x = F.sigmoid(x) * 100
        return x.transpose(1,2), tuple([state1[0].detach(), state1[1].detach()]), extra
