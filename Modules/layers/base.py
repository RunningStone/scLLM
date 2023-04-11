"""
base class for all layers to make it easy to customize layers

all of the peft methods need to be done in this level rather than in architecture level
"""

import torch
import torch.nn as nn
from scLLM.Modules.ops.base import BasicOps

class BaseLayers(nn.Module):
    def __init__(
        self,
        ops = BasicOps(),
    ):
        super().__init__()
        self.ops = ops
