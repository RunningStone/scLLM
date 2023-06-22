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