
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
# modality-specific embedding
class MS_Emb(nn.Module,):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()

        self.t_embedding = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        ) 


        self.s_embedding = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_src, s_src):
        t_src = self.t_embedding(t_src)
        s_src = self.s_embedding(s_src)

        return t_src, s_src
    
# fusion module for diffierent modalities
class Emb_Fusion(nn.Module):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()

        self.t_fusion = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size, bias=False),
        ) 


        self.s_fusion = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size, bias=False),
        )


    def forward(self, t_src, s_src):
        t_src = self.t_fusion(t_src)
        s_src = self.s_fusion(s_src)

        return t_src, s_src

# spatio-temporal transformer encoder
class ST_TR(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer) -> None:
        super().__init__()
        self.d_model  = hidden_size 

        self.pe = PositionalEncoding(hidden_size)
        t_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.) 
        self.t_tr = TransformerEncoder(t_layer, num_layer)

        s_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.)
        self.s_tr = TransformerEncoder(s_layer, num_layer)


    def forward(self, t_src, s_src):
        t_psrc = self.pe(t_src)
        t_out = self.t_tr(t_psrc)
        t_g = t_out.amax(dim=1)

        s_out = self.s_tr(s_src)
        s_g = s_out.amax(dim=1)

        out = torch.cat([t_g,s_g], dim=1)    
        return out