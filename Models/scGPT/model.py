import torch
from torch import nn
from scLLM.Modules.utils import  exists, cast_tuple
from scLLM.Modules.layers.gene_encoder import Gene2VecPositionalEmbedding
from scLLM.Modules.Performer import Performer
from scLLM.Modules.utils import  Always
from scLLM.Modules.init import APEX_AVAILABLE
from scLLM.Models.scGPT.paras import scGPT_para
from scLLM.Modules.FlashTransformer import FlashTransformer


class scGPT_model(nn.Module):
    def __init__(self,paras:scGPT_para) -> None:
        super().__init__()
        self.paras = paras
        self.net = FlashTransformer(
            ntoken=paras.ntoken,
            d_model=paras.d_model,
            nhead = paras.nhead,
            d_hid = paras.d_hid,
            nlayers = paras.nlayers,

            # optional
            nlayers_cls=paras.nlayers_cls,
            n_cls=paras.n_cls,
            vocab=paras.vocab,
            dropout=paras.dropout,

            pad_token=paras.pad_token,
            pad_value=paras.pad_value,

            do_mvc=paras.do_mvc,
            do_dab=paras.do_dab,

            use_batch_labels=paras.use_batch_labels,
            num_batch_labels=paras.num_batch_labels,
            domain_spec_batchnorm=paras.domain_spec_batchnorm, 

            input_emb_style=paras.input_emb_style,
            n_input_bins=paras.n_input_bins,

            cell_emb_style=paras.cell_emb_style,
            mvc_decoder_style = paras.mvc_decoder_style,

            ecs_threshold=paras.ecs_threshold,
            explicit_zero_prob=paras.explicit_zero_prob,
            use_fast_transformer=paras.use_fast_transformer,
            fast_transformer_backend=paras.fast_transformer_backend,

            pre_norm=paras.pre_norm,

            # specified ops
            ops_class_name=paras.ops_class_name,
            ops_class_para= paras.ops_class_para,
            # for forward which following part to get results or targets
            CLS= paras.CLS,
            CCE= paras.CCE,
            MVC= paras.MVC,
            ECS = paras.ECS,
            do_sample = paras.do_sample,
        )
        

    def forward(self,
                input_gene_ids,input_values,
                src_key_padding_mask,batch_labels=None):
                        
        output = self.net(input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels)
        
        return output