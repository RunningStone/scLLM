import torch
from scLLM.Models.scRankNet.paras import scRank_para
from scLLM.Trainer.paras import Trainer_para
from scLLM.Models.scBERT.utils import CosineAnnealingWarmupRestarts
from scLLM.Modules.customised_loss import RankNetLoss

####################################################################
#         trainer paras
####################################################################
trainer_para = Trainer_para()
trainer_para.project = None # project name
trainer_para.entity= None # entity name
trainer_para.exp_name = "scBERT_finetune_fast_" # experiment name
#-----> dataset
trainer_para.task_type = None # "classification","regression"
trainer_para.class_nb = None # number of classes
trainer_para.batch_size = 1 # batch size
trainer_para.shuffle = False
trainer_para.num_workers =0
trainer_para.additional_dataloader_para = {}
#-----> model
trainer_para.model_name: str = "scBERT" #"scFormer","scBERT"
trainer_para.pre_trained:str = None # pre-trained ckpt location

#-----> optimizer and loss
trainer_para.optimizers:list = [torch.optim.Adam,
                       ] # list of optimizer
trainer_para.optimizer_paras:list =[
        {"lr":1e-4},# for first optimizer
    ] # list of optimizer paras dict
"""

"""
trainer_para.schedulers:list = [CosineAnnealingWarmupRestarts,] # list of scheduler
trainer_para.scheduler_paras:list = [
    # for CosineAnnealingWarmupRestarts
    # aim 100 epochs or higher
    {"first_cycle_steps":15,
    "cycle_mult":2,
    "max_lr":1e-4,
    "min_lr":1e-6,
    "warmup_steps":5,
    "gamma":0.9,
    }] # list of scheduler paras dict

trainer_para.loss= RankNetLoss # loss function class but not create instance here
                     # create instance in pl_basic class with Trainer_para.loss()
trainer_para.loss_para= None # loss function paras

trainer_para.clip_grad= 0.5#int(1e6) # clip gradient

#-----> training
trainer_para.max_epochs= 10

#-----> metrics
trainer_para.metrics_names = ["accuracy","f1_score","precision"] # list of metrics name
trainer_para.metrics_paras = None

#-----> pytorch lightning paras
trainer_para.trainer_output_dir = None # output dir for pytorch lightning
# additional pytorch lightning paras
trainer_para.additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                "accumulate_grad_batches":10, # less batch size need accumulated grad
                "accelerator":"auto",#accelerator='gpu', devices=1,
            }

trainer_para.with_logger = "wandb" # "wandb",
trainer_para.wandb_api_key = None # wandb api key

trainer_para.save_ckpt = True # save checkpoint or not
trainer_para.ckpt_folder:str = None
#debug: try to add formated name to ckpt
trainer_para.ckpt_format:str = "_{epoch:02d}-{accuracy_val:.2f}" # check_point format
trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                   "monitor":"accuracy_val",}
metrics_multiclass = True
trainer_para.metrics_paras = {
"classification":{
    "accuracy": {"average": "micro","multiclass":metrics_multiclass},
    "cohen_kappa": {},
    "f1_score": {"average": "macro","multiclass":metrics_multiclass},
    "recall": {"average": "macro","multiclass":metrics_multiclass},
    "precision": {"average": "macro","multiclass":metrics_multiclass},
    "specificity": {"average": "macro","multiclass":metrics_multiclass},

    "auroc": {"average": "macro","multiclass":metrics_multiclass},
    "roc": {"average": "macro","multiclass":metrics_multiclass},
    "confusion_matrix": {"normalize": "true","multiclass":metrics_multiclass},
},}

####################################################################
#         model paras
####################################################################
model_para = scRank_para()

model_para.model_name= "scRank"                   # model name
model_para.model_type = "direct" # ["direct","multiply"]
model_para.out_dim = 1
#----> paras for create model architecture
model_para.num_tokens=5+2                         # num of tokens
model_para.max_seq_len=16906+1#24447+1              # max length of sequence
model_para.dim=200                                # dim of tokens
model_para.depth=6                              # layers
model_para.heads=10
model_para.local_attn_heads = 0
model_para.g2v_position_emb = True
model_para.g2v_weight_loc = None
