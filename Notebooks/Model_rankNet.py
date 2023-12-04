import torch
#  debug overall setting
torch.autograd.set_detect_anomaly(True)
import sys
sys.path.append("/home/to/scLLM_workspace")
# datalocs 
###############################################################################################
#
###############################################################################################
from scLLM.Dataset.paras import Dataset_para
# define pre-processing by follow original implementation of scBERT
dataset_para = Dataset_para(
                            var_idx=None,
                            obs_idx='cell_type',#"pseudotimes",Emt_label
                            vocab_loc="vocab/json/path",#NEED TO ADD
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

data_path = "Data/path" #NEED TO ADD
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
from scLLM.Dataset.dataset import SCDatasetRankSample

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
from scLLM.Modules.customised_loss import RankNetLoss

import torch
import torch.nn as nn
from scLLM.Trainer.base import pl_basic
from scLLM.Trainer.paras import Trainer_para
from scLLM import logger
from scLLM.Models.scBERT.pl import pl_scBERT_rankNet

import torch
from scLLM.Models.scBERT.paras import scBERT_para
from scLLM.Trainer.paras import Trainer_para
from scLLM.Models.scBERT.utils import CosineAnnealingWarmupRestarts


####################################################################
#         trainer paras
####################################################################
from scLLM.Predefine.scBERR_rank import trainer_para,model_para


####################################################################
#         scBERT
####################################################################
#-----> project
trainer_para.project = "EMT-scBERT-Rank" # project name
trainer_para.entity= "shipan_work" # entity name
trainer_para.exp_name = "rank_net_as_classification" # experiment name
#-----> dataset
trainer_para.task_type = "classification" # "classification","regression"
trainer_para.class_nb = None # number of classes # NEED TO ADD
trainer_para.batch_size =1 # batch size
#-----> model
trainer_para.pre_trained = "ckpt/path" # NEED TO ADD
trainer_para.ckpt_folder = "ckpt/folder/path" # NEED TO ADD

#-----> training and opt
trainer_para.max_epochs= 200
trainer_para.scheduler_paras:list = [
    # for CosineAnnealingWarmupRestarts
    # aim 200 epochs or higher
    {"first_cycle_steps":30,
    "cycle_mult":2,
    "max_lr":5e-5,
    "min_lr":1e-6,
    "warmup_steps":10,
    "gamma":0.9,
    }] # list of scheduler paras dict

#-----> pytorch lightning paras
trainer_para.trainer_output_dir = "Output/path" # NEED TO ADD
trainer_para.wandb_api_key = "NEED TO ADD"

trainer_para.additional_pl_paras.update({"precision":"16"})#"amp_backend":"apex","precision":"bf16",

#-----> model paras
model_para.g2v_weight_loc = "gene2vec/path" # NEED TO ADD
model_para.class_nb = trainer_para.class_nb
model_para.drop = 0.1
#-----> peft paras
PEFT_name = "lora"
from scLLM.Modules.ops.lora import default_lora_para
lora_para = default_lora_para
lora_para.r = 64
lora_para.lora_alpha = 16
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

#--------> save model
peft.pl_model.trainer.save_checkpoint(trainer_para.ckpt_folder+trainer_para.exp_name+"_last.ckpt")