"""
Flash attention from https://github.com/HazyResearch/flash-attention 
required for scGPT
"""

import torch
import math

from typing import Dict, Mapping, Optional, Tuple, Any, Union
from functools import partial
from einops import  repeat,rearrange

import torch
from torch import nn, Tensor

import torch.cuda.amp as amp
from torch.cuda.amp import autocast
import torch.nn.functional as F

#-------> import from internal modules
from scLLM.Modules.utils import default,null_context
from scLLM.Modules.init import APEX_AVAILABLE


#simplified version of flash attention with pytorch 2.0 scaled_dot_product_attention
def FlashAtten_pytorch2_func( qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
                return_softmax, deterministic):
    scale_factor = 1 / math.sqrt(qkv.size(-1)) if softmax_scale is None else softmax_scale
    attn_mask = torch.ones(max_seqlen, max_seqlen, dtype=torch.bool).tril(diagonal=0) if causal else None
    attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf')) if attn_mask is not None and attn_mask.dtype==torch.bool else attn_mask
    
    # replace with pytorch 2.0 scaled_dot_product_attention
    qkv = qkv.unsqueeze(0) # nnz,h,d -> 1,nnz,h,d #scaled_dot_product_attention need to have a batch dimension
    attn_weight = torch.nn.functional.scaled_dot_product_attention(
        qkv[:,:, 0],qkv[:,:, 1], qkv[:,:, 2], attn_mask=attn_mask, dropout_p=dropout_p, 
        is_causal=causal,
    )
    attn_weight = attn_weight.squeeze(0) # 1,nnz,h,d -> nnz,h,d
    return attn_weight if not return_softmax else (attn_weight, None, None)  # softmax output might need adjustment
"""
# replaced version with original flash attention implementation code
class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
                return_softmax, deterministic):
        scale_factor = 1 / math.sqrt(qkv.size(-1)) if softmax_scale is None else softmax_scale
        attn_mask = torch.ones(max_seqlen, max_seqlen, dtype=torch.bool).tril(diagonal=0) if causal else None
        attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf')) if attn_mask is not None and attn_mask.dtype==torch.bool else attn_mask
        
        # replace with pytorch 2.0 scaled_dot_product_attention
        qkv = qkv.unsqueeze(0)
        attn_weight = torch.nn.functional.scaled_dot_product_attention(
            qkv[:,:, 0],qkv[:,:, 1], qkv[:,:, 2], attn_mask=attn_mask, dropout_p=dropout_p, 
            is_causal=causal,
        )

        ctx.save_for_backward(qkv, attn_weight)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.scale = scale_factor
        ctx.causal = causal
        return attn_weight if not return_softmax else (attn_weight, None, None)  # softmax output might need adjustment

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, attn_weight = ctx.saved_tensors
        dout = dout.contiguous()  
        
        # Calculate gradients. Gradients for query, key and value tensors might need adjustment.
        dq, dk, dv = torch.autograd.grad(attn_weight, (qkv[:,:, 0], qkv[:,:, 1], qkv[:,:, 2]), dout)
        
        return torch.stack((dq, dk, dv), dim=1), None, None, None, None, None, None, None
"""

def flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None,
                                       causal=False, return_attn_probs=False, deterministic=False):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        deterministic: bool. Whether or not to ensure deterministic execution.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    #return FlashAttnQKVPackedFunc.apply(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale,causal, return_attn_probs, deterministic)
    return FlashAtten_pytorch2_func(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale,causal, return_attn_probs, deterministic)



class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, 'b ... -> b (...)'), 0,
                            repeat(indices, 'z -> z d', d=second_dim)).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                  device=grad_output.device, dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, 'z -> z d', d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

index_first_axis = IndexFirstAxis.apply

class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
                             dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None
    
index_put_first_axis = IndexPutFirstAxis.apply

def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)

def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices), indices,
            cu_seqlens, max_seqlen_in_batch)

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        #print(qkv.shape)
        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                #print(qkv.shape,cu_seqlens.shape,max_s)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                #print(f"unpadded x with shape {x_unpad.shape},max_s is {max_s}")
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                #print(f"output_unpad shape after flash attn is {output_unpad.shape}")
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


