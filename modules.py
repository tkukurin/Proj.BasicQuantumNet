import gin
import string

import tensornetwork as tn

import torch
from torch import nn


def _mps_einsum(n):
  chars = string.ascii_lowercase
  lhs = [f'z{a}{b}' for a,b in zip(chars, chars[1:n+1])]
  return f'{",".join(lhs)}->z{lhs[-1][-1]}'


@gin.configurable
class Model(nn.Module):
  def __init__(self, hidden_dim, out_dim=2):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.clf = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      #nn.LayerNorm(hidden_dim),
      #nn.Dropout(p=0.4),
      nn.ReLU(),
      nn.Linear(hidden_dim, out_dim))

  def forward(self, s1, s2):
    #import pdb;pdb.set_trace()
    #ops1 = _mps_einsum(s1.shape[1])
    #ops2 = _mps_einsum(s2.shape[1])
    #print(ops1)

    s1 = torch.einsum('blwh->bwh', s1)  #torch.prod(s1, dim=1)
    s2 = torch.einsum('blwh->bwh', s2)  #torch.prod(s2, dim=1)
    return self.clf((s1+s2).sum(-1))
    #return self.clf((s1+s2).max(-1)[0])

