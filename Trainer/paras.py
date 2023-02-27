from typing import Any, Optional, Union
from functools import partial


class Trainer_para:
    # need to set when init
    #-----> dataset

    #-----> model
    model_name: str = None #"scFormer","scBERT"
    ckpt_loc: str = None   # checkpoint file location

    #-----> optimizer and loss
    optimizers:list = [] # list of optimizer
    optimizer_paras:list =[] # list of optimizer paras dict

    schedulers:list = [] # list of scheduler
    scheduler_paras:list = [] # list of scheduler paras dict

    loss: Any = None # loss function class but not create instance here
                     # create instance in pl_basic class with Trainer_para.loss()
    loss_para: dict = None # loss function paras
    #-----> training
    max_epochs: int = 100

    #-----> metrics

    #-----> functions
    def init_optimizers(self):
        """
        init optimizers
        """
        assert len(self.optimizers) >0
        
        self.opt_instances = []
        self.sch_instances = []
        # partial init opts
        for opt,opt_para in zip(self.optimizers,self.optimizer_paras):
            partial_opt = partial(opt, **opt_para)
            self.opt_instances.append(partial_opt)
        # partial init schs
        if self.schedulers is not None or len(self.schedulers) >0:
            assert len(self.schedulers) == len(self.optimizers)
            for sch,sch_para in zip(self.schedulers,self.scheduler_paras):
                partial_sch = partial(sch, **sch_para)
                self.sch_instances.append(partial_sch)
        
        
