"""
Tokenize and pad a batch of data. 
Returns a list of tuple (gene_id, count).
"""
import random
import torch

def random_sample(data,label,cls_nb,device="cpu"):
    """
    Randomly sample a gene from the vocabulary.
    """
    rand_start = random.randint(0, data.shape[0]-1)
    full_seq = data[rand_start].toarray()[0]
    full_seq[full_seq > cls_nb] = cls_nb
    full_seq = torch.from_numpy(full_seq).long()
    full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
    seq_label = label[rand_start]
    return full_seq, seq_label