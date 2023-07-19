
import torch
import torch.nn as nn

from scLLM.Trainer.base import pl_basic
from scLLM.Trainer.paras import Trainer_para
from scLLM import logger
class pl_scGPT(pl_basic):
    def __init__(self,
                    trainer_paras:Trainer_para, # trainer para
                    model_paras, # model para
                    ):
            super(pl_scGPT,self).__init__(trainer_paras = trainer_paras,
                                           model_paras=model_paras)
            logger.info("init scGPT pytorch-lightning ...")
            self.create_model()
            self.configure_optimizers()

    def create_model(self):
        """
        create model instance
        """
        from scLLM.Models.scGPT.model import scGPT_model
        logger.info("init scGPT model...")
        logger.debug(f"model paras: {self.model_paras}")
        self.model = scGPT_model(self.model_paras)
        logger.info("init done...")

    def create_loss(self):
        """
        create loss instance
        """
        logger.info("create loss instance...")
        logger.debug(f"loss: {self.trainer_paras.loss}")
        assert self.trainer_paras.loss is not None or len(self.trainer_paras.additional_loss_fn)>0 
        if self.trainer_paras.loss is not None:
            if self.trainer_paras.loss_para is None:
                self.loss_fn = self.trainer_paras.loss()
            else:
                self.loss_fn = self.trainer_paras.loss(**self.trainer_paras.loss_para)
        if self.trainer_paras.additional_loss_fn is not None and len(self.trainer_paras.additional_loss_fn)>0:
            # 把self.trainer_paras.additional_loss_fn 中每个item key对应的value都实例化，然后注册为self.additional_loss_fn_{key} 形式
            for item in self.trainer_paras.additional_loss_fn:
                name = item["name"]
                loss_fn = item["fn"]
                paras = item["paras"]
                if paras is None:
                    loss_fn = loss_fn()
                else:
                    loss_fn = loss_fn(**paras)
                #保证name不重复，注册self.{name}
                assert not hasattr(self,f"{name}")
                setattr(self,f"{name}",loss_fn)

    def create_metrics(self,):
        logger.info("scGPT create metrics instance...")
        if self.model_paras.CLS:
            logger.info("classification task create metrics...")
            self.trainer_paras.init_metrics_factory()
            self.bar_metrics = self.trainer_paras.metrics_factory.metrics["metrics_on_bar"]
            self.valid_metrics = self.trainer_paras.metrics_factory.metrics["metrics_template"].clone(prefix = 'val_')
            self.train_metrics = self.trainer_paras.metrics_factory.metrics["metrics_template"].clone(prefix = 'train_')
        else:
            logger.warning("no metrics created...")

    def calc_loss(self,input_values,output_dict,target_values,batch_labels=None,celltype_labels=None):
        # criterion == mse_loss_fn
        # criterion_neg_log_bernoulli = zero_log_prob_loss_fn
        # criterion_cls == cls_loss_fn
        # criterion_dab == dab_loss_fn
        masked_positions = input_values.eq(self.model_paras.mask_value)  # the postions to predict
        loss = 0.0
        metrics_to_log = {}
        if self.model_paras.MLM:
            loss_mse = self.mse_loss_fn(
                output_dict["mlm_output"], target_values, masked_positions
            )
            loss = loss + loss_mse
            metrics_to_log = {"train/mse": loss_mse.item()}
        if self.model_paras.explicit_zero_prob:
            loss_zero_log_prob = self.zero_log_prob_loss_fn(
                output_dict["mlm_zero_probs"], target_values, masked_positions
            )
            loss = loss + loss_zero_log_prob
            metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
        if self.model_paras.CLS:
            loss_cls = self.cls_loss_fn(output_dict["cls_output"], celltype_labels)
            loss = loss + loss_cls
            metrics_to_log.update({"train/cls": loss_cls.item()})

            error_rate = 1 - (
                (output_dict["cls_output"].argmax(1) == celltype_labels)
                .sum()
                .item()
            ) / celltype_labels.size(0)
            metrics_to_log.update({"train/batch_cls_error_rate": error_rate})
        if self.model_paras.CCE:
            loss_cce = 10 * output_dict["loss_cce"]
            loss = loss + loss_cce
            metrics_to_log.update({"train/cce": loss_cce.item()})
        if self.model_paras.MVC:
            loss_mvc = self.mse_loss_fn(
                output_dict["mvc_output"], target_values, masked_positions
            )
            loss = loss + loss_mvc
            metrics_to_log.update({"train/mvc": loss_mvc.item()})
        if self.model_paras.MVC and self.model_paras.explicit_zero_prob:
            loss_mvc_zero_log_prob = self.zero_log_prob_loss_fn(
                output_dict["mvc_zero_probs"], target_values, masked_positions
            )
            loss = loss + loss_mvc_zero_log_prob
            metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
        if self.model_paras.ECS:
            loss_ecs = 10 * output_dict["loss_ecs"]
            loss = loss + loss_ecs
            metrics_to_log.update({"train/ecs": loss_ecs.item()})
        if self.model_paras.DAB:
            assert batch_labels is not None
            # try weighting and separate optimizer
            loss_dab = self.dab_loss_fn(output_dict["dab_output"], batch_labels)
            loss = loss + self.model_paras.dab_weight * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})
        
        return loss,metrics_to_log
    ####################################################################
    #              what into model and what out dict are defined
    ####################################################################
    def train_data_preprocess(self,batch):
        """
        data preprocess
        """
        #batch_data, label = batch
        raise NotImplementedError



    def train_post_process(self,metrics_to_log,output_dict,celltype_labels,loss):
        """
        post process
        """
        self.log_dict(metrics_to_log)
        
        if self.model_paras.CLS:
            output_values = output_dict["cls_output"]
            out = {
               "Y_hat":output_values,"label":celltype_labels,
               "loss":loss}
        else:
            out = {"loss":loss}
        return out

    def training_step(self, batch, batch_idx):
        #---->data preprocess
        input_gene_ids = batch["gene_ids"].to(self.model)#.to(device)
        input_values = batch["values"].to(self.model)#.to(device)
        target_values = batch["target_values"].to(self.model)#.to(device)
        batch_labels = batch["batch_labels"].to(self.model)#.to(device)
        celltype_labels = batch["celltype_labels"].to(self.model) if  self.model_paras.CLS else None
            
        src_key_padding_mask = input_gene_ids.eq(self.model_paras.vocab[self.model_para.pad_token])

        #---->forward step
        with torch.cuda.amp.autocast(enabled=self.model_paras.amp_flag):#enabled=amp_flag):
            #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):

            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.model_paras.use_batch_labels else None,
            )

            #---->loss step
            loss,metrics_to_log = self.calc_loss(input_values,output_dict,target_values,
                                batch_labels=batch_labels,celltype_labels=celltype_labels)

        #---->post process
        out = self.train_post_process(metrics_to_log,output_dict,celltype_labels,loss)
        return out



    def val_data_preprocess(self,batch):
        """
        data preprocess
        """
        raise NotImplementedError

    def val_post_process(self,metrics_to_log,output_dict,celltype_labels,loss):
        """
        post process
        """
        self.log_dict(metrics_to_log)
        
        if self.model_paras.CLS:
            output_values = output_dict["cls_output"]
            out = {
               "Y_hat":output_values,"label":celltype_labels,
               "loss":loss}
        else:
            out = {"loss":loss}
        return out

    def validation_step(self, batch, batch_idx):
        #---->data preprocess
        input_gene_ids = batch["gene_ids"].to(self.model)#.to(device)
        input_values = batch["values"].to(self.model)#.to(device)
        target_values = batch["target_values"].to(self.model)#.to(device)
        batch_labels = batch["batch_labels"].to(self.model)#.to(device)
        celltype_labels = batch["celltype_labels"].to(self.model) if  self.model_paras.CLS else None
            
        src_key_padding_mask = input_gene_ids.eq(self.model_paras.vocab[self.model_para.pad_token])

        #---->forward step
        with torch.cuda.amp.autocast(enabled=self.model_paras.amp_flag):#enabled=amp_flag):
            #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):

            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.model_paras.use_batch_labels else None,
            )

            #---->loss step
            loss,metrics_to_log = self.calc_loss(input_values,output_dict,target_values,
                                batch_labels=batch_labels,celltype_labels=celltype_labels)

        #---->post process
        out = self.val_post_process(metrics_to_log,output_dict,celltype_labels,loss)
        return out
    ####################################################################
    #              what to log
    ####################################################################
    def log_val_metrics(self,outlist,bar_name:str):
        loss = self.collect_step_output(key="loss",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(loss, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
        
    def log_train_metrics(self,outlist,bar_name:str):
        loss = self.collect_step_output(key="loss",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(loss, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)    



        