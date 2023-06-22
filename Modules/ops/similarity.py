import torch.nn as nn

from scLLM.Modules.ops.base import BaseLayers

class CosineSimilarity_div_temp(nn.Module,BaseLayers):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp,**kwargs):
        nn.Module.__init__(self,)
        BaseLayers.__init__(self,**kwargs)
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp