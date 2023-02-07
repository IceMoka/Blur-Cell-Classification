import torch
import torch.nn as nn


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=1)[:, 1]
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=(0.3, 0.05, 0.3, 0.1, 0.2, 0.05), gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        log_pt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        log_pt = log_pt.view(-1)
        ce_loss = -log_pt
        pt = torch.exp(log_pt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
