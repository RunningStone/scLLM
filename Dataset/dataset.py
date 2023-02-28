"""
common dataset class for scLLM
"""
from torch.utils.data import DataLoader, Dataset
from scLLM.Dataset.tokenizer import random_sample
class SCDataset(Dataset):
    def __init__(self, data, label,cls_nb=19,device="cpu"):
        super().__init__()
        self.data = data
        self.label = label
        self.cls_nb = cls_nb
        self.device = device

    def __getitem__(self, index):
        full_seq, seq_label = random_sample(self.data, self.label,
                                    cls_nb=self.cls_nb,
                                    device=self.device)
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]