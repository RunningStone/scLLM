import argparse
from pathlib import Path
import torch
def train(task_name,code_loc,raw_data_loc,vocab_loc, model_ckpt,vocab_params, out_loc):
    # your original code here
    # 把scLLM的位置添加进system path保证可以import scLLM
    import sys
    sys.path.append(code_loc)

    # 数据集读取
    #---> 定义数据集参数
    from scLLM.Dataset.paras import Dataset_para

    # config follows scBERT model pre-processing requests
    dataset_para = Dataset_para(
        vocab_loc=vocab_loc,
        var_idx = None,#"genes.gene_short_name",
        obs_idx="pseudotimes",
        filter_gene_by_counts=False,
        filter_cell_by_counts=200,
        log1p=True,
        log1p_base=2,

        cls_nb=5,
        data_layer_name="X_log1p",
        label_key = "pseudotimes",
        #binarize=None, # not binarize use original label
    )

    # -----> 读取数据集 create
    """
    from scLLM.Dataset.Reader import scReader
    data_reader = scReader(dataset_para)
    # init vocab from default file loc or from list/dict given as params
    data_reader.init_vocab()
    #load anndata
    data_reader.load_adata(loc = raw_data_loc,translate=False)

    #data_reader.preprocess()

    trainset,valset,weights = data_reader.postprocess()

    """
    #-----> 读取数据集 dill
    import dill
    with open(raw_data_loc,"rb") as f:
        trainset,valset,weights,label_dict = dill.load(f)

    assert label_dict is not None
    dataset_para.cls_nb = len(label_dict)
    # 输出数据集信息
    print("trainset size: ",len(trainset))
    print("valset size: ",len(valset)) if valset is not None else None
    print("label_dict: ",label_dict)


    import torch
    import numpy as np

    from scLLM.Predefine.scBERT_classification import model_para,trainer_para

    #-----> project
    trainer_para.project = "EMT_LLM_LoRA" # project name
    trainer_para.entity= "shipan_work" # entity name
    trainer_para.exp_name = task_name # experiment name
    #-----> dataset
    trainer_para.task_type = "classification" # "classification","regression"
    trainer_para.class_nb = dataset_para.cls_nb # number of classes
    trainer_para.batch_size =1 # batch size
    #-----> model
    trainer_para.pre_trained = model_ckpt#"/Users/shipan/Documents/workspace_scLLM/pre_trained/EMT_scBERT/FFT_emt_easy_scLLM_scBERT-3_cls3_LR5e-05_77.ckpt"#"//main/PAN/Exp03_scLLM/pre_trained/scBERT/panglao_pretrain.pth"
    trainer_para.ckpt_folder = str(Path(model_ckpt).parent)+"/" #"/Users/shipan/Documents/workspace_scLLM/pre_trained/EMT_scBERT/"

    #-----> pytorch lightning paras
    #accuracy_val
    trainer_para.max_epochs = 100 # max epochs
    trainer_para.save_ckpt = True # save checkpoint or not
    trainer_para.ckpt_format:str = "_{epoch:02d}-{accuracy_val:.2f}" # check_point format # 注意这里我们没有用f-string，而是留下了未格式化的模板字符串
    trainer_para.ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                   "monitor":"accuracy_val",
                   "mode":"max",}
    trainer_para.trainer_output_dir = "/home/pan/Experiments/EXPs/scLLM_workspace/Temp/" 
    trainer_para.wandb_api_key = "1266ad70f8bf7695542bf9a2d0dec8748c52431c"
    #trainer_para.additional_pl_paras.update({"amp_backend":"apex","precision":"16"})#"amp_backend":"apex","precision":"bf16"
    #amp_backend="apex"

    #-----> scBERT model paras
    model_para.g2v_weight_loc = vocab_params#"/Users/shipan/Documents/workspace_scLLM/pre_trained/scBERT/gene2vec_16906_200.npy"

    #-----> peft paras
    PEFT_name = "lora"
    from scLLM.Modules.ops.lora import default_lora_para
    lora_para = default_lora_para
    lora_para.r = 1
    lora_para.lora_alpha = 1
    lora_para.enable_lora = True
    #########################################################################
    #            get pl model          #
    #########################################################################
    from scLLM.Models.scBERT.pl import pl_scBERT
    pl_model = pl_scBERT(trainer_paras=trainer_para,model_paras=model_para)

    #--------> change the model to PEFT model
    from scLLM.Models.PEFT import get_peft
    peft = get_peft(pl_model,PEFT_name,lora_para)
    del pl_model
    # change output layer
    from scLLM.Modules.layers.out_layer import scBERT_OutLayer

    peft.load_model(original_ckpt = trainer_para.pre_trained)
    #-----> specify lora trainable params
    peft.set_trainable()
    peft.pl_model.model.to_out = scBERT_OutLayer(in_dim=model_para.max_seq_len,
                            dropout=0., 
                            h_dim=128, 
                            out_dim=trainer_para.class_nb,)
    # 对所有to_out的参数进行优化
    peft.pl_model.model.to_out.requires_grad_(True)

    from torch.utils.data.sampler import WeightedRandomSampler
    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    weights = [1.0 / len(trainset) for i in range(len(trainset))]
    trainsampler = WeightedRandomSampler(weights, len(weights))
    # 根据数据集的类别分布，给每个样本赋予一个权重，使得每个类别的样本被抽到的概率相同
    weights = [1.0 / len(valset) for i in range(len(valset))]
    valsampler = WeightedRandomSampler(weights, len(weights))

    #--------> get dataloader
    from torch.utils.data import DataLoader
    trainloader = DataLoader(trainset, batch_size=trainer_para.batch_size, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=trainer_para.batch_size, sampler=valsampler)

    peft.pl_model.build_trainer()
    #with autocast():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        peft.pl_model.trainer.fit(peft.pl_model,trainloader,valloader)

    #--------> save model
    peft.pl_model.trainer.save_checkpoint(trainer_para.ckpt_folder+trainer_para.exp_name+"_last.ckpt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Task1.')
    parser.add_argument('--task_name', type=str, help='Name of task')

    parser.add_argument('--code_loc', type=str, help='Location of source code')
    parser.add_argument('--raw_data_loc', type=str, help='Location of data')
    parser.add_argument('--vocab_loc', type=str, help='Location of model vocab')
    parser.add_argument('--model_ckpt', type=str, help='Location of model checkpoint')
    parser.add_argument('--vocab_params', type=str, help='Location of vocab mapping embedding')
    parser.add_argument('--out_loc', type=str, help='Output location')

    args = parser.parse_args()

    train(args.task_name,args.code_loc,args.raw_data_loc,args.vocab_loc, args.model_ckpt,args.vocab_params, args.out_loc)
