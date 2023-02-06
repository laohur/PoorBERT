
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer

# https://github.com/jjljkjljk/SimCSE-Chinese/blob/main/model.py

def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def simcse_sup_loss(y_pred, device, lamda=0.05):
    """
    有监督损失函数
    """
    similarities = F.cosine_similarity(
        y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    row = torch.arange(0, y_pred.shape[0], 3)
    col = torch.arange(0, y_pred.shape[0])
    col = col[col % 3 != 0]

    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / lamda

    y_true = torch.arange(0, len(col), 2, device=device)
    loss = F.cross_entropy(similarities, y_true)
    return loss


if __name__ == '__main__':
    y_pred = torch.rand((30, 16))
    loss = simcse_sup_loss(y_pred, 'cpu', lamda=0.05)
    print(loss)
