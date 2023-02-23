import torch
import random
import pytorch_lightning as pl

from scLLM.Trainer.paras import Trainer_para

class pl_basic(pl.LightningModule):
    def __init__(self,
                 trainer_paras:Trainer_para,# trainer para
                model_paras, # model para
                ):
        super(pl_basic,self).__init__()
        """
        A basic class for different PL protocols:
        """
        print("init basic part...")
        #----> paras
        self.trainer_paras = trainer_paras
        self.model_paras = model_paras
        #----> create model
        self.model = None

        #----> create loss
        self.create_loss()

        #----> create metrics
        self.create_metrics()

    def create_model(self):
        """
        create model instance
        """
        pass
    
    def create_loss(self):
        """
        create loss instance
        """
        if self.trainer_paras.loss_para is None:
            self.trainer_paras.loss_fn = self.trainer_paras.loss()
        else:
            self.trainer_paras.loss_fn = self.trainer_paras.loss(**self.trainer_paras.loss_para)

    def configure_optimizers(self):
        """
        create optimizer and scheduler

        return [optimizer],[scheduler]
        """
        opt_list = [opt(self.model.parameters()) \
                        for opt in self.trainer_paras.opt_instances]
        if self.trainer_paras.sch_instances is not None \
                or len(self.trainer_paras.sch_instances) >0:
            sch_list = [sch(opt) for sch,opt in \
                            zip(self.trainer_paras.sch_instances,opt_list)]
            return opt_list,sch_list
        else:
            return opt_list

    def create_metrics(self,):
        """
        create metrics instance
        """
        pass

    def re_shuffle(self,is_shuffle:bool=True):
        #---->random, if shuffle data, change seed
        if is_shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def collect_step_output(self,key,out,dim=None):
        if dim is None:
            return torch.cat([x[key] for x in out])
        else:
            return torch.cat([x[key] for x in out],dim=dim)
        

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_epoch_end(self, training_step_outputs):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError

