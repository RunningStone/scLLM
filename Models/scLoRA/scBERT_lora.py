"""
very same model with scBERT but with lora training

# This sets requires_grad to False for all parameters without the string "lora_" in their names
from scLLM.Modules.lora import mark_only_lora_as_trainable,lora_state_dict
mark_only_lora_as_trainable(model)

# for save state_dict 
# ===== After =====
torch.save(lora_state_dict(model), checkpoint_path)

# for load state_dict
# Load the pretrained checkpoint first
model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
# Then load the LoRA checkpoint
model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)

"""
import torch
from torch import nn
from scLLM.Modules.utils import  exists, cast_tuple
from scLLM.Modules.gene_encoder import Gene2VecPositionalEmbedding
from scLLM.Modules.Performer import Performer_lora
from scLLM.Modules.utils import  Always
from scLLM.Modules.init import APEX_AVAILABLE
from scLLM.Models.scLoRA.paras import scBERT_lora_para

#------> for lora training
from scLLM.Modules.lora import LoRALinear,LoRAEmbedding
class PerformerLM_lora(nn.Module):
    def __init__(
        self,
        paras:scBERT_lora_para,
    ):
        super().__init__()
        self.paras = paras

        local_attn_heads = cast_tuple(self.paras.local_attn_heads)

        self.max_seq_len = self.paras.max_seq_len
        #lora replace nn.Embedding
        self.token_emb = LoRAEmbedding(self.paras.num_tokens, self.paras.dim) 
        
        if self.paras.g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(self.paras.g2v_weight_loc,self.paras.max_seq_len)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(self.paras.emb_dropout)

        self.performer = Performer_lora(self.paras.dim, 
                                   self.paras.depth, 
                                   self.paras.heads, 
                                   self.paras.dim_head, 
                                   local_attn_heads, 
                                   self.paras.local_window_size, 
                                   self.paras.causal, 
                                   self.paras.ff_mult, 
                                   self.paras.nb_features, 
                                   self.paras.feature_redraw_interval, 
                                   self.paras.reversible, 
                                   self.paras.ff_chunks, 
                                   self.paras.generalized_attention, 
                                   self.paras.kernel_fn, 
                                   self.paras.use_scalenorm, 
                                   self.paras.use_rezero, 
                                   self.paras.ff_glu, 
                                   self.paras.ff_dropout, 
                                   self.paras.attn_dropout, 
                                   self.paras.cross_attend, 
                                   self.paras.no_projection, 
                                   self.paras.auto_check_redraw, 
                                   self.paras.qkv_bias)
        self.norm = nn.LayerNorm(self.paras.dim)
        #lora replace nn.Linear
        self.to_out = LoRALinear(self.paras.dim, self.paras.num_tokens) if not self.paras.tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)

        if output_attentions:
            x, attn_weights = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x
            #@ means matrix multiplication __matmul__()
            return x @ self.token_emb.weight.t()