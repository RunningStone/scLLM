"""
basically same as scBERT but change some part with lora
"""
import torch
from scLLM.Trainer.paras import Trainer_para
from scLLM.Models.scBERT.pl import pl_scBERT
from scLLM import logger
from scLLM.Modules.lora import mark_only_lora_as_trainable,lora_state_dict


class pl_scBERT_lora(pl_scBERT):
    def __init__(self,
                    trainer_paras:Trainer_para, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scBERT_lora,self).__init__(trainer_paras = trainer_paras,
                                           model_paras=model_paras)
            
            assert model_paras.lora_para is not None, "lora_para is None"

            logger.info("init as LoRA version pytorch-lightning ...")

    def set_lora_trainable(self,):
        mark_only_lora_as_trainable(self.model)

    def load_model(self,original_ckpt:str,lora_ckpt_loc:str,map_device:str="cuda"):
        # Load the pretrained checkpoint first
        self.model.load_state_dict(torch.load(original_ckpt,map_location=map_device), strict=False)
        # Then load the LoRA checkpoint
        self.load_lora(lora_ckpt_loc,map_device)
    
    def load_lora(self,lora_ckpt_loc:str,map_device:str="cuda"):
        # Then load the LoRA checkpoint
        self.model.load_state_dict(torch.load(lora_ckpt_loc,map_location=map_device), strict=False)

    def save_lora(self,lora_path:str):
        # save lora separately and manually
        torch.save(lora_state_dict(self.model), lora_path)

