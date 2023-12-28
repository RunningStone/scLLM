"""
common dataset class for scLLM
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Union, Optional


############################################################
#               Dataset for scBERT
############################################################
from scLLM.Dataset.utils import random_sample
class SCDataset(Dataset):
    def __init__(self, data, label,cls_nb=19):
        super().__init__()
        self.data = data
        self.label = label
        self.cls_nb = cls_nb

    def __getitem__(self, index):
        full_seq, seq_label = random_sample(self.data, self.label,
                                    cls_nb=self.cls_nb,
                                    )
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]
    
############################################################
#               Dataset for scGPT
############################################################
# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    

############################################################
#               Dataset for scBERT Rank
############################################################
import torch
import numpy as np
import random
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
    
############################################################
#               Dataset for scBERT simple sample
############################################################
class SCDatasetRankSimpleSample(Dataset):
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
        self.part1 = [0]
        self.part2 = [4]

    def to_tensor(self,full_seq):
        full_seq = full_seq.toarray()[0]
        full_seq[full_seq > self.thresh] = self.thresh
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq

    def __getitem__(self, index):
        # 选择第一个样本
        # 从self.part1中随机选择一个类别
        class1 = random.choice(self.part1)
        # 从该类别的索引中随机选择一个样本
        index1 = random.choice(self.indices_per_class[class1])
        full_seq1 = self.data[index1]
        full_seq1 = self.to_tensor(full_seq1)
        cls_label1 = self.label[index1]
        # 从self.part2中随机选择一个类别
        class2 = random.choice(self.part2)
        # 从该类别的索引中随机选择一个样本
        index2 = random.choice(self.indices_per_class[class2])
        #print(index2)
        full_seq2 = self.data[index2]
        full_seq2 = self.to_tensor(full_seq2)
        cls_label2 = self.label[index2]

        return (full_seq1, full_seq2), (cls_label1, cls_label2)

    def set_fix_len(self, fix_len:int=5000):
        self.fix_len = fix_len

    def __len__(self):
        if self.fix_len is not None:
            return self.fix_len
        # 计算self.part1中所有类别的样本数量
        count_part1 = sum(len(self.indices_per_class[cls]) for cls in self.part1)
        # 计算self.part2中所有类别的样本数量
        count_part2 = sum(len(self.indices_per_class[cls]) for cls in self.part2)

        # 返回所有可能组合的数量
        return count_part1 * count_part2