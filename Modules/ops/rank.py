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