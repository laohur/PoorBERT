import torch
import torch.nn as nn
import numpy as np


class BombMaker(nn.Module):
    """
    mask in dims
    https://github.com/dayyass/pytorch-ner/blob/32e8cdae22037babc038e340bd6a6eedaa0ae92d/pytorch_ner/nn_modules/dropout.py
    """

    def __init__(self, p: float,dim:int=-1):
        super(BombMaker, self).__init__()
        self.p=p
        self.dim=dim
        self.spatial_dropout = nn.Dropout2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose (1, self,self.dim)
        x = self.spatial_dropout(x)
        x = x.transpose (1, self,self.dim)
        return x
