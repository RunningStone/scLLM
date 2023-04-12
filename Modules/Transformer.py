import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from scLLM.Modules.layers.base import BaseLayers
class Transformer(nn.Module):
    def __init__(
        self,
        #---------> transformer encoder
        dim: int,       # dimension
        heads: int,     # heads
        dim_head: int,  # dimension of head
        depth: int,     # depth of transformer

        dropout: float = 0.5,
        **kwargs
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        encoder_layers = nn.TransformerEncoderLayer(
            dim, heads, dim_head, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, depth)
        

    def forward(
        self,
        encoded_input: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            encoded_input (:obj:`Tensor`): shape [batch_size, seq_len, embsize]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]


        Returns:
            dict of output Tensors.
        """
        transformer_output = self.transformer_encoder(encoded_input, 
                                                      src_key_padding_mask=src_key_padding_mask)
        #transformer_output  # (batch, seq_len, embsize)
        return transformer_output

    

