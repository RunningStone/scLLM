"""
performer for scBERT MODEL
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Callable

# import from scLLM
from scLLM.Modules.utils import  exists, cast_tuple,get_module_device,find_modules
from scLLM.Modules.utils import  ReZero, Chunk


from scLLM.Modules.Norm import PreLayerNorm, PreScaleNorm
from scLLM.Modules.reversible import ReversibleSequence, SequentialSequence
from scLLM.Modules.fastAttention import SelfAttention,FastAttention
from scLLM.Modules.feedForward import FeedForward

# performer

class Performer(nn.Module):
    def __init__(
        self,
        dim:int,                                # dimension
        depth:int,                              # layers
        heads:int,                              # heads
        dim_head:int,                           # dim of head
        local_attn_heads:int = 0,               # num of local attention heads, (heads - local_attn_heads) is num of global performers
        local_window_size:int = 256,            # window size of local attention
        causal:bool = False,                     # autoregressive or not
        ff_mult:int = 4,                        # dim of intermediate features after attention / dim of input features
        nb_features:int = None,                 # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head ?? what is random feature ??
        feature_redraw_interval:int = 1000,     # how frequently to redraw the projection matrix, the more frequent, the slower the training
        reversible:bool = False,                 # reversible layers, from Reformer (save memory)
        ff_chunks:int = 1,                      # chunk feedforward layer, from Reformer
        generalized_attention:bool = False,      # defaults to softmax approximation, but can be set to True for generalized attention ?? what is generalized attention ??
        kernel_fn:Callable = nn.ReLU(),          # the kernel function to be used, if generalized attention is turned on, defaults to Relu
        use_scalenorm:bool = False,              # use scale norm, from 'Transformers without Tears' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        use_rezero:bool = False,                 # use Rezero or not, from 'Rezero is all you need' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        ff_glu:bool = False,                     # use GLU (Gated Linear Units) variant for feedforward
        ff_dropout:float = 0.,                    # feedforward dropout
        attn_dropout:float = 0.,                  # post-attention dropout
        cross_attend:bool = False,               # cross_attend(decoder)
        no_projection:bool = False,              # with final linear projection or not
        auto_check_redraw:bool = True,           # check redraw projections for all attention layers or not
        qkv_bias:bool = True,                    # qkv with bias or not
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))
            # if no need cross_attend(decoder), begin next cycle
            if not cross_attend:
                continue
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # ((True, False), (True, False), (True, False), (True, False), (True, False), (True, False))
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions = False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, output_attentions = output_attentions, **kwargs)



