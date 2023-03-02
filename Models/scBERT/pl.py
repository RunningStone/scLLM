
import torch.nn as nn
from scLLM.Trainer.base import pl_basic
from scLLM.Trainer.paras import Trainer_para
from scLLM import logger
class pl_scBERT(pl_basic):
    def __init__(self,
                    trainer_paras:Trainer_para, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scBERT,self).__init__(trainer_paras = trainer_paras,
                                           model_paras=model_paras)
            logger.info("init scBERT pytorch-lightning ...")
            self.create_model()
            self.configure_optimizers()

    def create_model(self):
        """
        create model instance
        """
        from scLLM.Models.scBERT.model import PerformerLM
        logger.info("init scBERT model...")
        logger.debug(f"model paras: {self.model_paras}")
        self.model = PerformerLM(self.model_paras)
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
        return data, label

    def train_post_process(self,logits,label,loss):
        """
        post process
        """
        #---->metrics step
        softmax = nn.Softmax(dim=-1)
        prob = softmax(logits)
        final = prob.argmax(dim=-1)
        out = {"logits":logits,"Y_prob":prob,
               "Y_hat":final,"label":label,
               "loss":loss}
        return out

    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0)
        return data, label

    def val_post_process(self,logits,label,loss):
        """
        post process
        """
        #---->metrics step
        softmax = nn.Softmax(dim=-1)
        prob = softmax(logits)
        final = prob.argmax(dim=-1)
        out = {"logits":logits,"Y_prob":prob,
               "Y_hat":final,"label":label,
               "loss":loss}
        return out

    ####################################################################
    #              what to log
    ####################################################################
    def log_val_metrics(self,outlist,bar_name:str):
        probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        
    def log_train_metrics(self,outlist,bar_name:str):
        probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)    



        