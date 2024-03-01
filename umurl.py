
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
