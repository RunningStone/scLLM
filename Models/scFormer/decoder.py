import torch
from torch import Tensor,nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Bernoulli
from typing import Optional,Any,Union
from scLLM.Modules.gene_decoders import MVCDecoder,ExprDecoder,ClsDecoder,AdversarialDiscriminator
from scLLM.Modules.utils import Similarity
class Decoders(nn.Module):
    def __init__(self,
                 dim: int,# dimension of the model
                 #----> for cls decoder
                 n_cls: int,# number of classes
                 nlayers_cls: int,# number of layers in the cls decoder
                 #---->  decoder common paras
                 explicit_zero_prob: bool = False,# explicit zero probability
                 use_batch_labels: bool = False,# whether to use batch labels
                 #----> for mvc decoder
                 do_mvc: bool = False,# whether to use mvc decoder
                 mvc_decoder_style: str = "inner product",# style of mvc decoder
                 #----> for dab decoder
                 do_dab: bool = False,# whether to use dab decoder
                 num_batch_labels: Optional[int] = None,# number of batch labels
                 #----> for ecs 
                 ecs_threshold: float = 0.3,# threshold for elastic cell similarity
                 ):
        super().__init__()
        # get paras
        self.explicit_zero_prob = explicit_zero_prob
        self.use_batch_labels = use_batch_labels
        self.do_mvc = do_mvc
        self.cur_gene_token_embs = None
        self.ecs_threshold = ecs_threshold
        # init models
        self.sim = Similarity(temp=0.5)  # TODO: auto set temp
        self.expr_decoder = ExprDecoder(
            dim,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        self.cls_decoder = ClsDecoder(dim, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                dim,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                dim,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
    def init_weights(self, initrange: float = 0.1):
        self.expr_decoder.bias.data.zero_()
        self.expr_decoder.weight.data.uniform_(-initrange, initrange)

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
        self.CLS = CLS
        self.CCE = CCE
        self.MVC = MVC
        self.ECS = ECS
        self.do_sample = do_sample

    
    def forward(self,transformer_output,cell_emb,
                cell2=None,
                batch_emb=None):
        output = {}
        mlm_output = self.expr_decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and self.do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]


        if self.CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if self.CCE:
            cell1 = cell_emb

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            # loss calc outside
            output["cos_sim"] = cos_sim
            output["labels"] = labels
        if self.MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and self.do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if self.ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        
        return output

