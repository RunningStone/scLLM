import torch
#  debug overall setting
torch.autograd.set_detect_anomaly(True)

# datalocs 
###############################################################################################
#
###############################################################################################
from scLLM.Dataset.paras import Dataset_para
# define pre-processing by follow original implementation of scBERT
dataset_para = Dataset_para(
                            var_idx=None,
                            obs_idx='cell_type',#"pseudotimes",Emt_label
                            vocab_loc="/home/shi/WorkSpace/projects/scLLM_workspace/pre_trained/scBERT/vocab_16k.json",
                            filter_gene_by_counts=False,
                            filter_cell_by_counts=200,
                            log1p=True,
                            log1p_base=2,
                            #--> specify the tokenizer
                            tokenize_name="scBERT",#"scGPT",
                            label_key='cell_type',#"pseudotimes",Emt_label
                            binarize=None,#"equal_instance",
                            batch_label_key="batch_id",
                            test_size = None,
                            #-->specified in pre-trained model
                            cls_nb=5,
                            max_len=16906, #48292+2, # vocab size+2 or gene size+2
                            )
print(dataset_para)


# init preprocessor
from scLLM.Dataset.Reader import scReader
screader = scReader(dataset_para=dataset_para)
# init vocab
screader.init_vocab()

data_path = "/home/shi/WorkSpace/projects/scLLM_workspace/data/Eloise/dataset/EMT_Cook/TrVal_dataset_PC_TGFb_GTlabel5.pkl"
data_path = "/home/shi/WorkSpace/projects/scLLM_workspace/data/Eloise/dataset/EMT_Cook/TrVal_dataset_Breast_TGFb_GTlabel.pkl"
import dill
with open(data_path,"rb") as f:
  out=dill.load(f)
[trainset,valset,_,label_dict]=out


# 输出数据集信息
print("trainset size: ",len(trainset))
print("valset size: ",len(valset)) if valset is not None else print("No val part")

print(f"dataset label map: {label_dict}")
resort_label = False
print (f" resort label later : {resort_label}")
###############################################################################################
# # Dataset define
###############################################################################################


import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Union, Optional
import random

from scLLM.Dataset.dataset import SCDataset
class SCDatasetRankSample(Dataset):
    def __init__(self, data, label,thresh):
        super().__init__()
        self.data = data
        self.label = label
        # 使用torch.unique来确定类别数量
        unique_labels = torch.unique(label)
        self.cls_nb = len(unique_labels)
        self.thresh = thresh
        # 按类别分组样本的索引
        self.indices_per_class = {i: np.where(label == i)[0] for i in range(self.cls_nb)}

    def to_tensor(self,full_seq):
        full_seq = full_seq.toarray()[0]
        full_seq[full_seq > self.thresh] = self.thresh
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq

    def __getitem__(self, index):
        # 选择第一个样本
        cls_label1 = self.label[index]
        full_seq1 = self.data[index]
        full_seq1 = self.to_tensor(full_seq1)
        # 选择第二个样本，确保其类别不同于第一个样本
        class_type = [i.item() for i in torch.unique(self.label)]
        different_classes = list(set(class_type) - {cls_label1.item()})
        #print(different_classes)
        cls_label2_idx = random.choice(different_classes)
        #print(cls_label2_idx)
        index2 = random.choice(self.indices_per_class[cls_label2_idx])
        #print(index2)
        full_seq2 = self.data[index2]
        full_seq2 = self.to_tensor(full_seq2)
        cls_label2 = self.label[index2]

        return (full_seq1, full_seq2), (cls_label1, cls_label2)

    def __len__(self):
        return self.data.shape[0]

def re_sort_labels(original_labels,original_dict=label_dict,rank_dict={'0d': 0, '8h': 1, '1d': 2, '3d': 3, '7d': 4}):
    # 创建反向映射字典
    reverse_label_dict = {v: k for k, v in original_dict.items()}
    # 转换为字符串标签列表
    string_labels = [reverse_label_dict[label.item()] for label in original_labels]
    # 重新映射为新的数值标签
    new_numeric_labels = torch.tensor([rank_dict[label] for label in string_labels])
    return new_numeric_labels

if resort_label:
    new_label_dict = {'0d': 0, '8h': 1, '1d': 2, '3d': 3, '7d': 4}
    train_label =re_sort_labels(trainset.label)
    val_label =re_sort_labels(valset.label)
else:
    train_label = trainset.label
    val_label = valset.label

rank_trainset = SCDatasetRankSample(data= trainset.data, label=train_label,thresh=1)
rank_valset = SCDatasetRankSample(data= valset.data, label=val_label,thresh=1)


(d1,d2),(l1,l2) = next(iter(rank_trainset))
print(d1.shape,d2.shape)
print(l1,l2)
print(l1-l2)

###############################################################################################
#  Model define
###############################################################################################
import torch


def _ranknet_loss(logits_i, logits_j, label_i, label_j):
    # compute_probability(logits_i, logits_j)
    P_ij= torch.sigmoid(logits_i - logits_j)
    # compute loss
    sig1 = (label_i.detach() > label_j.detach()).float()
    sig2 = (label_i.detach() < label_j.detach()).float()
    loss = - sig1 * torch.log(P_ij) - sig2 * torch.log(1 - P_ij)
    return loss

import torch
import torch.nn as nn

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, logits_i, logits_j, labels_i, labels_j):
        loss = _ranknet_loss(logits_i, logits_j, labels_i, labels_j)
        return loss


import torch
import torch.nn as nn
from scLLM.Trainer.base import pl_basic
from scLLM.Trainer.paras import Trainer_para
from scLLM import logger

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



import torch
from scLLM.Models.scBERT.paras import scBERT_para
from scLLM.Trainer.paras import Trainer_para
from scLLM.Models.scBERT.utils import CosineAnnealingWarmupRestarts


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
    {"first_cycle_steps":5,
    "cycle_mult":2,
    "max_lr":1e-4,
    "min_lr":1e-6,
    "warmup_steps":2,
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

trainer_para.metrics_paras = {
"classification":{
    "accuracy": {"average": "micro","multiclass":True},
    "cohen_kappa": {},
    "f1_score": {"average": "macro","multiclass":True},
    "recall": {"average": "macro","multiclass":True},
    "precision": {"average": "macro","multiclass":True},
    "specificity": {"average": "macro","multiclass":True},

    "auroc": {"average": "macro","multiclass":True},
    "roc": {"average": "macro","multiclass":True},
    "confusion_matrix": {"normalize": "true","multiclass":True},
},}

####################################################################
#         model paras
####################################################################
model_para = scBERT_para()
model_para.num_tokens=5+2                         # num of tokens
model_para.max_seq_len=16906+1#24447+1              # max length of sequence
model_para.dim=200                                # dim of tokens
model_para.depth=6                              # layers
model_para.heads=10
model_para.local_attn_heads = 0
model_para.g2v_position_emb = True
model_para.g2v_weight_loc = None


####################################################################
#         scBERT
####################################################################
# get predefined parameters
from scLLM.Predefine.scBERT_classification import model_para
#-----> project
trainer_para.project = "EMT-scBERT-Rank" # project name
trainer_para.entity= "shipan_work" # entity name
trainer_para.exp_name = "rank_net_as_classification" # experiment name
#-----> dataset
trainer_para.task_type = "classification" # "classification","regression"
trainer_para.class_nb = 2 # number of classes
trainer_para.batch_size =1 # batch size
#-----> model
trainer_para.pre_trained = "/home/shi/WorkSpace/projects/scLLM_workspace/pre_trained/scBERT/panglao_pretrain.pth" #"/content/drive/MyDrive/Data/Pan_DATA/scLLM/pre_trained/finetuned/emt_easy_scLLM_scBERT-2_cls3_LR5e-05_0.77.ckpt" #%[]
trainer_para.ckpt_folder = "/home/shi/WorkSpace/projects/scLLM_workspace/pre_trained/EMT_scBERT"

#-----> pytorch lightning paras
trainer_para.trainer_output_dir = "/home/shi/WorkSpace/projects/scLLM_workspace/Temp"
trainer_para.wandb_api_key = None


#-----> model paras
model_para.g2v_weight_loc = "/home/shi/WorkSpace/projects/scLLM_workspace/pre_trained/scBERT/gene2vec_16906_200.npy"
model_para.class_nb = trainer_para.class_nb
model_para.drop = 0.1
#-----> peft paras
PEFT_name = "lora"
from scLLM.Modules.ops.lora import default_lora_para
lora_para = default_lora_para
lora_para.r = 1
lora_para.lora_alpha = 1
lora_para.enable_lora = True


####################################################################
#         init pl model
####################################################################
pl_model = pl_scBERT_rankNet(trainer_paras=trainer_para,model_paras=model_para)

#--------> change the model to PEFT model
from scLLM.Models.PEFT import get_peft
peft = get_peft(pl_model,PEFT_name,lora_para)



peft.load_model(original_ckpt = trainer_para.pre_trained)
#-----> specify lora trainable params
peft.set_trainable()
# change output layer
from scLLM.Modules.layers.out_layer import scBERT_OutLayer
peft.pl_model.model.to_out = scBERT_OutLayer(in_dim=model_para.max_seq_len,
                        dropout=model_para.drop,
                        h_dim=128,
                        out_dim=1,)
# 对所有to_out的参数进行优化
peft.pl_model.model.to_out.requires_grad_(True)


from torch.utils.data.sampler import WeightedRandomSampler
# 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
weights = [1.0 / len(rank_trainset) for i in range(len(rank_trainset))]
trainsampler = WeightedRandomSampler(weights, len(weights))
# 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
weights = [1.0 / len(rank_valset) for i in range(len(rank_valset))]
valsampler = WeightedRandomSampler(weights, len(weights))

#--------> get dataloader
from torch.utils.data import DataLoader
trainloader = DataLoader(rank_trainset, batch_size=trainer_para.batch_size, sampler=trainsampler)
valloader = DataLoader(rank_valset, batch_size=trainer_para.batch_size, sampler=valsampler)

#detect_anomaly=True,overfit_batches=0.01

#peft.pl_model.trainer_paras.additional_pl_paras.update({"detect_anomaly":True})#"overfit_batches":0.1,
peft.pl_model.build_trainer()

#with torch.autograd.set_detect_anomaly(True):
peft.pl_model.trainer.fit(peft.pl_model,trainloader,valloader)