from typing import Optional, Union
import torch
from torch import Tensor,nn

from scLLM.Modules.utils import tensorlist2tensor
from scLLM.Modules.Transformer import Transformer
from scLLM.Models.scFormer.paras import scFormer_para
from scLLM.Models.scFormer.encoder import Encoder
from scLLM.Models.scFormer.decoder import Decoders

class scFormer(nn.Module):
    def __init__(self, paras:scFormer_para):
        super().__init__()
        self.paras = paras
        self.encoder = Encoder(
            ntoken=paras.ntoken,
            dim=paras.dim,
            vocab=paras.vocab,
            pad_token=paras.pad_token,
            input_emb_style=paras.input_emb_style,
            cell_emb_style=paras.cell_emb_style,
            dropout=paras.dropout,
            n_input_bins=paras.n_input_bins,
            pad_value=paras.pad_value,
            use_batch_labels=paras.use_batch_labels,
            num_batch_labels=paras.num_batch_labels,
            domain_spec_batchnorm=paras.domain_spec_batchnorm,
            )
        self.decoder = Decoders(
            dim=paras.dim,
            n_cls=paras.n_cls,
            nlayers_cls=paras.nlayers_cls,
            explicit_zero_prob=paras.explicit_zero_prob,
            use_batch_labels=paras.use_batch_labels,
            do_mvc=paras.do_mvc,
            mvc_decoder_style=paras.mvc_decoder_style,
            do_dab=paras.do_dab,
            num_batch_labels=paras.num_batch_labels,
            ecs_threshold=paras.ecs_threshold,
        )

        self.transformer = Transformer(
            dim=paras.dim,
            heads=paras.heads,
            dim_head=paras.dim_head,
            depth=paras.depth,
            dropout=paras.dropout,
        )

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        #self.transformer.init_weights()

    def set_output(self,
                   CLS: bool = False,
                    CCE: bool = False,
                    MVC: bool = False,
                    ECS: bool = False,
                    do_sample: bool = False
                    ):
        """
        Args:
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.
        """
        self.decoder.set_output(CLS,CCE,MVC,ECS,do_sample)

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
        # encoder
        encoder_output,src_key_padding_mask, batch_emb = self.encoder(src, values, src_key_padding_mask, batch_labels)
        # transformer
        transformer_output = self.transformer(encoder_output, src_key_padding_mask)
        cell_emb = self.encoder._get_cell_emb_from_layer(transformer_output,values)
        # decoder
        if self.decoder.CCE:
            transformer_output2 = self.transformer(encoder_output, src_key_padding_mask)
            cell2 = self.encoder._get_cell_emb_from_layer(transformer_output2)
        else:
            transformer_output2 = None
            cell2 = None
        if self.decoder.MVC: self.decoder.cur_gene_token_embs = self.encoder.cur_gene_token_embs
        decoder_output = self.decoder(transformer_output,cell_emb,
                                        cell2=cell2,
                                        batch_emb=batch_emb)
        return decoder_output
    

    #########################################################################
    #   Inferece part
    #########################################################################
    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene == "batch-wise":
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            output_dict = self(
                input_gene_ids.repeat(batch_size, 1),
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            output_values = output_dict["mlm_output"]
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        else:
            input_gene_ids_list = []
            input_values = []
            input_pert_flags = []

            tmp_ = ori_gene_values != 0
            for row_i in range(batch_size):
                input_gene_id = tmp_[row_i].nonzero().flatten()
                input_gene_ids_list.append(input_gene_id)
                input_values.append(ori_gene_values[row_i][input_gene_id])
                input_pert_flags.append(pert_flags[row_i][input_gene_id])

            input_gene_ids = tensorlist2tensor(
                input_gene_ids_list, pad_value=self.pad_token_id
            )
            input_values = tensorlist2tensor(input_values, pad_value=self.pad_value)
            input_pert_flags = tensorlist2tensor(
                input_pert_flags, pad_value=self.pert_pad_id
            )

            src_key_padding_mask = input_gene_ids.eq(self.pad_token_id)
            output_dict = self(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            output_values = output_dict["mlm_output"]
            pred_gene_values = torch.zeros_like(ori_gene_values)
            for row_i in range(batch_size):
                input_gene_id = input_gene_ids_list[row_i]
                pred_gene_values[row_i, input_gene_id] = output_values[row_i][
                    : len(input_gene_id)
                ]
        return pred_gene_values