"""
implement 
- fast attention and fast self-attention for Performer
- self-attention_lora for Performer_lora
"""


import math

from functools import partial
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F



#-------> import from internal modules
from scLLM.Modules.utils import default, exists, empty
from scLLM.Modules.init import APEX_AVAILABLE
from scLLM.Modules.layers.gene_encoder import apply_rotary_pos_emb

#from scLLM.Modules.basic_block.lora import LoRALinear,LoRAEmbedding
from scLLM.Modules.layers.base import BaseLayers


############################################################################################################
#               self Attention with fast attention module
############################################################################################################
class SelfAttention(BaseLayers):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert dim % heads == 0, 'dimension must be divisible by number of heads'

        # init ops module
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = self.ops.FastAttention(dim_head, nb_features, causal = causal,
                                                      generalized_attention = generalized_attention,
                                                        kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = self.ops.LocalAttention(window_size = local_window_size, 
                                                  causal = causal, autopad = True, 
                                                  dropout = dropout, look_forward = int(not causal), 
                                                  rel_pos_emb_config = (dim_head, local_heads))\
                                                                         if local_heads > 0 else None

        self.to_q = self.ops.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = self.ops.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = self.ops.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = self.ops.Linear(inner_dim, dim)
        self.dropout = self.ops.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, output_attentions = False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k, = apply_rotary_pos_emb(q, k, pos_emb)

            if output_attentions:
                out, attn_weights = self.fast_attention(q, k, v, output_attentions)
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)     # combine attn_out and cross_attn_out, here we have only attn_out, that means this line does nothing
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if output_attentions:
            return self.dropout(out), attn_weights
        else:
            return self.dropout(out)