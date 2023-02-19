from typing import Optional,Any,Union
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from tqdm import trange
from scLLM.Modules.gene_encoder import GeneNNEncoder,\
        ContinuousValueEncoder,CategoryValueEncoder,\
        BatchLabelEncoder

from scLLM.Modules.Norm import DomainSpecificBatchNorm1d

class Encoder(nn.Module):
    def __init__(self, 
                ntoken, 
                dim:int,  # embedding dimension of model
                vocab:Any,      # vocab of padding token
                pad_token: str = "<pad>", # padding token idx
                input_emb_style:str = "continuous", #how to embed input values[continuous,category]
                cell_emb_style: str = "cls", # how to get cell embedding
                dropout:float=0.5, #dropout rate
                #----> for category value encoder
                n_input_bins:Optional[int] = None,  # input dim.number of bins for input value
                pad_value: int = 0, 
                #----> for batch label encoder
                use_batch_labels: bool = False, 
                num_batch_labels: Optional[int] = None, # input dim of batch labels
                #----> for norm
                domain_spec_batchnorm: Union[bool, str] = False, # domain specific batchnorm
                ):
        super(Encoder, self).__init__()
        # get paras from config
        self.input_emb_style = input_emb_style
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.cell_emb_style = cell_emb_style
        # check paras 
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneNNEncoder(ntoken, dim, padding_idx=vocab[pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(dim, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, dim, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, dim)

        if domain_spec_batchnorm:
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                dim, num_batch_labels, affine=use_affine
            )
        else:
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(dim)

        # initialize the weights for gene encoder
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)


    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if self.domain_spec_batchnorm:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        else:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        return total_embs, src_key_padding_mask
        #output = self.transformer_encoder(
        #    total_embs, src_key_padding_mask=src_key_padding_mask
        #)
        #return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len]
            values: Tensor, shape [N, seq_len]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        outputs = []
        N = src.size(0)
        device = next(self.parameters()).device
        for i in trange(0, N, batch_size):
            output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            if output_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def forward(self,
                src: Tensor,
                values: Tensor,
                src_key_padding_mask: Tensor,
                batch_labels: Optional[Tensor] = None):
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
        Returns:
            dict of output Tensors.
        """

        total_embs, src_key_padding_mask = self._encode(
            src, values, src_key_padding_mask, batch_labels
        )
        # (batch, embsize)
        batch_emb = self.batch_encoder(batch_labels)  if self.use_batch_labels else None # 

        return total_embs, src_key_padding_mask, batch_emb