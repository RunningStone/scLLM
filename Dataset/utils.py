import random
import torch

def mask_list(original_list,mask):
    """
    Mask a list with a mask
    :param original_list: list to be masked
    :param mask: mask with True/False values [True, False, True, ...]
    :return: masked list
    """
    return [x for x, m in zip(original_list, mask) if m]

def random_sample(data,label,cls_nb):
    """
    Randomly sample a gene from the vocabulary.
    """
    rand_start = random.randint(0, data.shape[0]-1)
    full_seq = data[rand_start].toarray()[0]
    full_seq[full_seq > cls_nb] = cls_nb
    full_seq = torch.from_numpy(full_seq).long()
    full_seq = torch.cat((full_seq, torch.tensor([0])))
    seq_label = label[rand_start]
    return full_seq, seq_label