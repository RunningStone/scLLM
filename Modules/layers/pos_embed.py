import math
import numpy as np
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
# sinusoidal positional embeddings
from scLLM import logger
from scLLM.Modules.layers.base import BaseLayers

#########################################################################################
#            rotary positional embeddings
#########################################################################################
# rotary positional embedding helpers for scBERT

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# rotary positional embedding helpers for scGPT

class RotaryPositionalEncoding(nn.Module,BaseLayers):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
                 **kwargs):
        super().__init__()
        BaseLayers.__init__(self,**kwargs)
        self.dropout = self.ops.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        logger.debug("RotaryPositionalEncoding initialised")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

#########################################################################################
#            absolute positional embeddings
#########################################################################################
# positional embeddings

class AbsolutePositionalEmbedding(nn.Module,BaseLayers):

    def __init__(self, dim, max_seq_len,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.emb = self.ops.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
    



