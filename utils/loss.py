import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        Args:
            alpha: 类别权重平衡系数 (0.5 表示正负样本权重相等)
            gamma: 聚焦参数 (越大越关注困难样本)
            reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 模型的原始输出 Logits [Batch, 1] (无 Sigmoid)
        # targets: 真实标签 [Batch, 1] (0 或 1)

        # 1. 计算标准的 BCE With Logits Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2. 计算预测概率 pt (用于衡量样本的难易程度)
        # pt = exp(-bce_loss) 这是一个数学上的恒等变换
        pt = torch.exp(-bce_loss)

        # 3. Focal Loss 公式: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function():
    # 返回默认配置的 Focal Loss
    return FocalLoss(alpha=0.5, gamma=2.0)