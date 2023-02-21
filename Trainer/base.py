import torch
import random
import pytorch_lightning as pl

class pl_basic(pl.LightningModule):
    def __init__(self):
        super(pl_basic,self).__init__()
        """
        A basic class for different PL protocols:
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
