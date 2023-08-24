import torch
import numpy as np
import logging
from scLLM import logger
logger.setLevel(logging.DEBUG)

# get predefined parameters
from scLLM.Predefine.scBERT_classification import model_para,trainer_para
#-----> project
trainer_para.project = "EMT_LLM" # project name
trainer_para.entity= "shipan_work" # entity name
trainer_para.exp_name = trainer_para.exp_name + "EMT" # experiment name
#-----> dataset
trainer_para.task_type = "classification" # "classification","regression"
trainer_para.class_nb = 5 # number of classes
trainer_para.batch_size =1 # batch size
#-----> model
trainer_para.pre_trained = "//main/PAN/Exp03_scLLM/pre_trained/scBERT/panglao_pretrain.pth"
trainer_para.ckpt_folder = "//main/PAN/Exp03_scLLM/pre_trained/finetuned/"

#-----> pytorch lightning paras
trainer_para.trainer_output_dir = "//main/PAN/Exp05_scLLM_PEFT/Temp/" 
trainer_para.wandb_api_key = "Your API key"


#-----> model paras
model_para.g2v_weight_loc = "//main/PAN/Exp03_scLLM/pre_trained/scBERT/gene2vec_16906_200.npy"


vocab_loc = "//main/PAN/Exp03_scLLM/pre_trained/scBERT/vocab_gene2vec_16906.pkl"
adata_loc = "//main/PAN/Exp03_scLLM/data/Eloise/emt_easy_scLLM.h5ad"

#########################################################################
#            get pl model          #
#########################################################################
from scLLM.Models.scBERT.pl import pl_scBERT
pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)
#########################################################################
#            get data from original raw format          #
#########################################################################
import pickle
with open(vocab_loc, "rb") as f:
    vocab = pickle.load(f)

from scLLM.Dataset.preprocessor import Preprocessor
from scLLM.Dataset.paras import Dataset_para
# define pre-processing by follow original implementation of scBERT
dataset_para = Dataset_para(gene_vocab=vocab,
                            filter_gene_by_counts=False,
                            filter_cell_by_counts=200,
                            log1p=True,
                            log1p_base=2,
                            )

preprocess = Preprocessor(dataset_para,trainer_para=trainer_para)
preprocess.load_adata(adata_loc)
data = preprocess.to_data(data_type="log1p")
label,class_weight = preprocess.to_label(
                          label_key="pseudotimes",
                          binarize="equal_instance",
                          bin_nb=trainer_para.class_nb,)

#########################################################################
#            load  pre-trained model and change output layer             #
#########################################################################
import torch
pre_model = torch.load(trainer_para.pre_trained)#, map_location=pl_model.model.device)
pl_model.model.load_state_dict(pre_model["model_state_dict"])

# change output layer
from scLLM.Modules.layers.out_layer import Identity
pl_model.model.to_out = Identity(in_dim=model_para.max_seq_len,
                        dropout=0., 
                        h_dim=128, 
                        out_dim=trainer_para.class_nb,)
#model.to(device)
for param in pl_model.model.parameters():
    param.requires_grad = False
for param in pl_model.model.norm.parameters():
    param.requires_grad = True
for param in pl_model.model.performer.net.layers[-1].parameters():
    param.requires_grad = True


#########################################################################
#           init dataloader with splited data and labels             #
#########################################################################
print(class_weight)
class_num = np.unique(label.numpy(), return_counts=True)[1].tolist()
print(class_num)
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2023)

idx_tr,idx_val = next(iter(sss.split(data, label)))
data_train, label_train = data[idx_tr], label[idx_tr]
data_val, label_val = data[idx_val], label[idx_val]

from torch.utils.data.sampler import WeightedRandomSampler
weights_train = [class_weight[label_train[i]] for i in range(label_train.shape[0])]
sampler_train = WeightedRandomSampler(torch.DoubleTensor(weights_train), len(weights_train))
weights_val = [class_weight[label_val[i]] for i in range(label_val.shape[0])]
sampler_val = WeightedRandomSampler(torch.DoubleTensor(weights_val), len(weights_val))


from scLLM.Dataset.dataset import SCDataset
train_dataset = SCDataset(data_train, label_train,cls_nb=trainer_para.class_nb)
val_dataset = SCDataset(data_val, label_val,cls_nb=trainer_para.class_nb)
train_loader = preprocess.to_dataloader(train_dataset, sampler=sampler_train)
val_loader = preprocess.to_dataloader(val_dataset,sampler=sampler_val)

#########################################################################

pl_model.build_trainer()
pl_model.trainer.fit(pl_model,train_loader,val_loader)
