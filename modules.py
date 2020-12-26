import gin
import string

import tensornetwork as tn

import torch
from torch import nn


class Model(nn.Module):
  def __init__(self, hidden_dim, vocab, out_dim=2):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.sos = nn.Parameter(torch.randn(hidden_dim))
    self.embed = torch.nn.Embedding(
      len(vocab), hidden_dim*hidden_dim, padding_idx=vocab.get(vocab.PAD))
    self.clf = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, out_dim))

  def forward(self, s1, s2):
    B, L1 = s1.shape
    B, L2 = s2.shape
    W, H = self.hidden_dim, self.hidden_dim

    s1 = self.embed(s1).reshape(B, L1, W, H)
    s2 = self.embed(s2).reshape(B, L2, W, H)

    s1 = torch.einsum('w,blwh->blh', self.sos, s1)
    s2 = torch.einsum('w,blwh->blh', self.sos, s2)

    s1 = torch.einsum('blh->bh', s1)
    s2 = torch.einsum('blh->bh', s2)

    return self.clf(s1 + s2)

