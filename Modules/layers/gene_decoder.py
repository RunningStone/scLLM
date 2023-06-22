import torch
import torch.nn as nn
from torch import Tensor
from typing import  Union, Dict

from scLLM.Modules.sequence.reversible import grad_reverse
from scLLM.Modules.layers.base import BaseLayers


############################################################################################################
#           Decoders for scGPT model
############################################################################################################
class ClsDecoder(nn.Module,BaseLayers):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
        **kwargs,
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        # module list
        self._decoder = self.ops.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(self.ops.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(self.ops.LayerNorm(d_model))
        self.out_layer = self.ops.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module,BaseLayers):
    """
    copy from scGPT/scgpt/model/model.py and modified
    Decoder for the masked value prediction for cell embeddings.

    There are actually three ways of making this, all start with gene_embs -> query_vecs,
    and then:
    1. cell_emb x W x query vecs.
       This one makes the most sense, since in the query space, the query look at
       different dimensions of cel_emb and sync them. This one has explicit interaction.
    2. FC([cell_emb, query_vecs]).
       This one has the benifit to have smaller query_vecs and makes them like bottle
       neck layer. For example 64 dims.
    3. FC(cell_emb + query_vecs).

    **NOTE**:
    1. it is important to make gene query vectors directly from the input
    gene embs. Because have to make sure there is no value information mixed in,
    and that is the only place to get the raw gene embs.
    2. Bare in mind to avoid short cut for the model to just predict
    value form the query. Make sure predict based on the cell_emb.
    3. Guess it will be better to use sigmoid for the query vecs.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = self.ops.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = self.ops.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = self.ops.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = self.ops.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = self.ops.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = self.ops.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = self.ops.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = self.ops.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = self.ops.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module,BaseLayers):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        # module list
        self._decoder = self.ops.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(self.ops.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(self.ops.LayerNorm(d_model))
        self.out_layer = self.ops.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class ExprDecoder(nn.Module,BaseLayers):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = self.ops.Sequential(
            self.ops.Linear(d_in, d_model),
            self.ops.LeakyReLU(),
            self.ops.Linear(d_model, d_model),
            self.ops.LeakyReLU(),
            self.ops.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = self.ops.Sequential(
                self.ops.Linear(d_in, d_model),
                self.ops.LeakyReLU(),
                self.ops.Linear(d_model, d_model),
                self.ops.LeakyReLU(),
                self.ops.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

