"""
for fine-tuning the output layer
"""
import torch
import torch.nn as nn

from scLLM.Modules.layers.base import BaseLayers


class Identity(BaseLayers):
    def __init__(self,in_dim, dropout = 0., h_dim = 100, out_dim = 10,**kwargs):
        super(Identity, self).__init__(**kwargs)
        self.conv1 = self.ops.Conv2d(1, 1, (1, 200))
        self.act = self.ops.ReLU()
        self.fc1 = self.ops.Linear(in_features=in_dim, out_features=512, bias=True)
        self.act1 = self.ops.ReLU()
        self.dropout1 = self.ops.Dropout(dropout)
        self.fc2 = self.ops.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = self.ops.ReLU()
        self.dropout2 = self.ops.Dropout(dropout)
        self.fc3 = self.ops.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x