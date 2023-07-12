"""

"""
import math
import numpy as np
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
# sinusoidal positional embeddings
from typing import Optional, Tuple
from scLLM import logger
from scLLM.Modules.layers.base import BaseLayers
#########################################################################################
#            gene 2 vec positional embeddings
#########################################################################################
# Gene2Vec used in scBERT model
class Gene2VecPositionalEmbedding(nn.Module,BaseLayers):
    def __init__(self, gene2vec_weight:str=None, max_seq_len:int=16907,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        logger.debug("Gene2VecPositionalEmbedding initialised")
        if gene2vec_weight is not None:
            gene2vec_weight = np.load(gene2vec_weight)
        else:
            max_seq_len = max_seq_len -1 
            # original paper use gene2vec with 16906x200
            # this is only for loading model
            gene2vec_weight = np.random.randn(max_seq_len, 200)
        # shape: (16906+1, 200) added channel in the end
        gene2vec_weight = np.concatenate((gene2vec_weight, 
                                        np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = self.ops.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

#########################################################################################
#            gene encoders for scGPT
#########################################################################################

class GeneNNEncoder(nn.Module,BaseLayers):
    # NN gene encoder, used in scGPT
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        **kwargs
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        logger.debug(f"GeneNNEncoder initialised with {num_embeddings}, dim {embedding_dim}, padding_idx {padding_idx}")
        self.embedding = self.ops.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = self.ops.LayerNorm(embedding_dim)
        logger.debug("GeneNNEncoder initialised")

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ContinuousValueEncoder(nn.Module,BaseLayers):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.dropout = self.ops.Dropout(p=dropout)
        self.linear1 = self.ops.Linear(1, d_model)
        self.activation = self.ops.ReLU()
        self.linear2 = self.ops.Linear(d_model, d_model)
        self.norm = self.ops.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module,BaseLayers):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        **kwargs
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.embedding = self.ops.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = self.ops.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module,BaseLayers):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        **kwargs
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.embedding = self.ops.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = self.ops.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x