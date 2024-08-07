# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, smoothing=0.01, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        self.off_value = smoothing
        self.on_value = 1. - smoothing

    def forward(self, predict, target):

        predict = predict.float().reshape(-1)
        target = torch.where(target > 0.5, self.on_value, self.off_value)

        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)- (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCEFocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smoothing=0.01):
        super(BCEFocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.off_value = smoothing
        self.on_value = 1. - smoothing

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0]
        input = input.float().reshape(-1)
        # input = torch.sigmoid(input)  # sigmoide获取概率

        target = torch.where(target > 0.5, self.on_value, self.off_value)
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

#
# def py_sigmoid_focal_loss(pred,
#                           target,
#                           weights=None,
#                           gamma=2.0,
#                           alpha=0.25,
#                           reduction='mean',
#                           avg_factor=None):
#     pred_sigmoid = pred.sigmoid()
#     target = target.type_as(pred)
#     pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
#     focal_weight = (alpha * target + (1 - alpha) *
#                     (1 - target)) * pt.pow(gamma)
#     loss = F.binary_cross_entropy_with_logits(
#         pred, target, reduction='none') * focal_weight
#     loss = weight_reduce_loss(loss, weights, reduction, avg_factor)
#     return loss


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """

    def __init__(
            self, smoothing=0.1, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('weights', weight)
        self.register_buffer('pos_weight', pos_weight)

        self.off_value = self.smoothing
        self.on_value = 1. - self.smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        x = x.float().reshape(-1)
        target = torch.where(target > 0.5, self.on_value, self.off_value)

        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)