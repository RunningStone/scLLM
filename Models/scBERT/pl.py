import torch
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
        label = label.squeeze(0)#.float()
        return data, label

    def train_post_process(self,logits,label,loss):
        """
        post process
        """
        if self.trainer_paras.task_type == "classification":
            #---->metrics step
            softmax = nn.Softmax(dim=-1)
            prob = softmax(logits)
            final = prob.argmax(dim=-1)
            out = {"logits":logits,"Y_prob":prob,
                "Y_hat":final,"label":label,
                "loss":loss}
            return out
        elif self.trainer_paras.task_type == "regression":
            #---->metrics step
            out = {"logits":logits.squeeze(0),
                   "label":label,
                "loss":loss}
            return out

    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        data, label = batch
        label = label.squeeze(0)#.type(torch.LongTensor) # <---- Here (casting)
        return data, label

    def val_post_process(self,logits,label,loss):
        """
        post process
        """
        #---->metrics step
        if self.trainer_paras.task_type == "classification":
            #---->metrics step
            softmax = nn.Softmax(dim=-1)
            prob = softmax(logits)
            final = prob.argmax(dim=-1)
            out = {"logits":logits,"Y_prob":prob,
                "Y_hat":final,"label":label,
                "loss":loss}
            return out
        elif self.trainer_paras.task_type == "regression":
            #---->metrics step
            out = {"logits":logits.squeeze(0),
                   "label":label,
                "loss":loss}
            return out

    ####################################################################
    #              what to log
    ####################################################################
    def log_val_metrics(self,outlist,bar_name:str):
        if self.trainer_paras.task_type == "classification":
            probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
            max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
            target = self.collect_step_output(key="label",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)
        elif self.trainer_paras.task_type == "regression":
            logits = self.collect_step_output(key="logits",out=outlist,dim=0)
            target = self.collect_step_output(key="label",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(logits, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.valid_metrics(logits.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)
        else:
            raise ValueError(f"task_type {self.trainer_paras.task_type} not supported")
        
    def log_train_metrics(self,outlist,bar_name:str):
        if self.trainer_paras.task_type == "classification":
            probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
            max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
            target = self.collect_step_output(key="label",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)    
        elif self.trainer_paras.task_type == "regression":
            logits = self.collect_step_output(key="logits",out=outlist,dim=0)
            target = self.collect_step_output(key="label",out=outlist,dim=0)
            #----> log part
            self.log(bar_name, self.bar_metrics(logits, target.squeeze()), 
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.train_metrics(logits.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)




#############################################################################################
#
#############################################################################################

class pl_scBERT_rankNet(pl_basic):
    def __init__(self,
                    trainer_paras:Trainer_para, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scBERT_rankNet,self).__init__(trainer_paras = trainer_paras,
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
    #              Train
    ####################################################################
    def training_step(self, batch, batch_idx):

        data_i, data_j, labels_i, labels_j = self.train_data_preprocess(batch)
        # 前向传播
        with torch.no_grad():
            logits_i = self.model(data_i)
        logits_j = self.model(data_j)

        # 计算 RankNet 损失
        loss = self.loss_fn(logits_i, logits_j, labels_i, labels_j)

        # 记录训练损失

        out = self.train_post_process(logits_i, logits_j, labels_i, labels_j,loss)
        return out

    def train_data_preprocess(self,batch):
        """
        data preprocess
        """
        # 解包批次数据
        (data_i, data_j), (labels_i, labels_j) = batch
        return data_i, data_j, labels_i, labels_j

    def train_post_process(self,logits_i, logits_j, labels_i, labels_j,loss):
        """
        post process
        """
        # detach avoid problems
        logits_i_d = logits_i.clone().detach()
        logits_j_d = logits_j.clone().detach()
        labels_i_d = labels_i.clone().detach()
        labels_j_d = labels_j.clone().detach()


        if self.trainer_paras.task_type == "classification":
            #---->metrics step
            prob = torch.sigmoid(logits_i_d - logits_j_d)
            final = prob.argmax(dim=-1)
            label = [1] if labels_i_d - labels_j_d > 0 else [0]
            device = prob.device
            label = torch.LongTensor(label).to(device)
            out = {"Y_prob":prob,
                "Y_hat":final,"label":label,
                "loss":loss,
                }
            return out
        else:
          raise NotImplementedError

    ####################################################################
    #              Val
    ####################################################################
    def validation_step(self, batch, batch_idx):
        #---->data preprocess
        data_i, data_j, labels_i, labels_j = self.train_data_preprocess(batch)
        # 前向传播
        with torch.no_grad():
            logits_i = self.model(data_i)
        logits_j = self.model(data_j)

        # 计算 RankNet 损失
        loss = self.loss_fn(logits_i, logits_j, labels_i, labels_j)


        #---->post process
        out = self.val_post_process(logits_i, logits_j, labels_i, labels_j,loss)
        return out

    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        # 解包批次数据
        (data_i, data_j), (labels_i, labels_j) = batch
        return data_i, data_j, labels_i, labels_j

    def val_post_process(self,logits_i, logits_j, labels_i, labels_j,loss):
        """
        post process
        """
        # detach avoid problems
        logits_i_d = logits_i.clone().detach()
        logits_j_d = logits_j.clone().detach()
        labels_i_d = labels_i.clone().detach()
        labels_j_d = labels_j.clone().detach()

        if self.trainer_paras.task_type == "classification":
            #---->metrics step
            prob = torch.sigmoid(logits_i_d - logits_j_d)
            final = prob.argmax(dim=-1)
            label = [1] if labels_i_d - labels_j_d > 0 else [0]
            device = prob.device
            label = torch.LongTensor(label).to(device)
            out = {"Y_prob":prob,
                "Y_hat":final,"label":label,
                }
            return out
        else:
          raise NotImplementedError


    ####################################################################
    #              what to log
    ####################################################################
    def log_val_metrics(self,outlist,bar_name:str):
        if self.trainer_paras.task_type == "classification":
            probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0).reshape(-1)
            max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0).reshape(-1)
            target = self.collect_step_output(key="label",out=outlist,dim=0).reshape(-1)
            print(probs.shape,probs.device,
                  max_probs.shape,max_probs.device,
                  target.shape,target.device)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()),
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)
        else:
            raise ValueError(f"task_type {self.trainer_paras.task_type} not supported")

    def log_train_metrics(self,outlist,bar_name:str):
        if self.trainer_paras.task_type == "classification":
            probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0).reshape(-1)
            max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0).reshape(-1)
            target = self.collect_step_output(key="label",out=outlist,dim=0).reshape(-1)
            print(probs.shape,probs.device,
                  max_probs.shape,max_probs.device,
                  target.shape,target.device)
            #----> log part
            self.log(bar_name, self.bar_metrics(probs, target.squeeze()),
                                prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)
        else:
            raise ValueError(f"task_type {self.trainer_paras.task_type} not supported")

