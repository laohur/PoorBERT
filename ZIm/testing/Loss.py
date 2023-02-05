
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


def simcse_unsup_loss(y_pred, temp=0.05):
    """无监督的损失函数
    NT-Xent loss  from SimCLR
    pytorch https://github.com/shuxinyin/SimCSE-Pytorch/blob/master/SimCSE/model.py
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    device=y_pred.device
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # [batch_size * 2, 1, 768] * [1, batch_size * 2, 768] = [batch_size * 2, batch_size * 2]
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp  # 相似度矩阵除以温度系数

    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)
