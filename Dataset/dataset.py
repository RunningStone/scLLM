"""
common dataset class for scLLM
"""
from torch.utils.data import DataLoader, Dataset
from scLLM.Dataset.tokenizer import random_sample
class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq, seq_label = random_sample(self.data, self.label)
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]