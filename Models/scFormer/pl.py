
import torch.nn as nn
from scLLM.Trainer.base import pl_basic
from scLLM.Trainer.paras import Trainer_para
from scLLM import logger
class pl_scFormer(pl_basic):
    def __init__(self,
                    trainer_paras:Trainer_para, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scFormer,self).__init__(trainer_paras = trainer_paras,
                                           model_paras=model_paras)
            logger.info("init scFormer pytorch-lightning ...")
            self.create_model()
            self.configure_optimizers()

    def create_model(self):
        """
        create model instance
        """
        from scLLM.Models.scFormer.model import scFormer
        logger.info("init scFormer model...")
        logger.debug(f"model paras: {self.model_paras}")
        self.model = scFormer(self.model_paras)
        logger.info("init done...")

    ####################################################################
    #              what into model and what out dict are defined
    ####################################################################
    def train_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0)
        if len(data) == 3:
            (gene_name,gene_value,name_key_padding_mask) = data
            return (gene_name,gene_value,name_key_padding_mask,None),label
        elif len(data) == 4:
            #(gene_name,gene_value,name_key_padding_mask,batch_labels) = data
            return data,label
        else:
            raise ValueError("data should only include 3 or 4 items as [gene_name,gene_value,name_key_padding_mask,batch_labels (optional)]]")


    def train_post_process(self,out,label,loss):
        """
        post process
        """
        if self.model.decoder.CLS:
            logits = out["cls_output"]
            #---->metrics step
            softmax = nn.Softmax(dim=-1)
            cls_prob = softmax(logits)
            out["cls_Y_prob"] = cls_prob
            Y_hat = cls_prob.argmax(dim=-1)
            out["cls_Y_hat"] = Y_hat
            out["cls_Y"] = label
            out["cls_loss"] = loss
        return out

    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0)
        if len(data) == 3:
            (gene_name,gene_value,name_key_padding_mask) = data
            return (gene_name,gene_value,name_key_padding_mask,None),label
        elif len(data) == 4:
            #(gene_name,gene_value,name_key_padding_mask,batch_labels) = data
            return data,label
        else:
            raise ValueError("data should only include 3 or 4 items as [gene_name,gene_value,name_key_padding_mask,batch_labels (optional)]]")


    def val_post_process(self,out,label,loss):
        """
        post process
        """
        #---->metrics step
        if self.model.decoder.CLS:
            logits = out["cls_output"]
            #---->metrics step
            softmax = nn.Softmax(dim=-1)
            cls_prob = softmax(logits)
            out["cls_Y_prob"] = cls_prob
            Y_hat = cls_prob.argmax(dim=-1)
            out["cls_Y_hat"] = Y_hat
            out["cls_Y"] = label
            out["cls_loss"] = loss
        return out

    ####################################################################
    #              what to log
    ####################################################################
    def log_val_metrics(self,outlist,bar_name:str):
        if self.model.decoder.CLS:
            probs = self.collect_step_output(key="cls_Y_prob",out=outlist,dim=0)
            max_probs = self.collect_step_output(key="cls_Y_hat",out=outlist,dim=0)
            target = self.collect_step_output(key="cls_Y",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)
        
    def log_train_metrics(self,outlist,bar_name:str):
        if self.model.decoder.CLS:
            probs = self.collect_step_output(key="cls_Y_prob",out=outlist,dim=0)
            max_probs = self.collect_step_output(key="cls_Y_hat",out=outlist,dim=0)
            target = self.collect_step_output(key="cls_Y",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)    



        