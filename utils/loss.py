# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    带有标签平滑的二分类交叉熵损失函数 (BCEWithLogitsLoss)。

    原理：
    将硬标签 [0, 1] 转换为软标签 [epsilon, 1 - epsilon]，
    其中 epsilon 是平滑因子。这能防止模型在训练时过度自信，从而减轻过拟合。

    参数:
        smoothing (float): 平滑因子 epsilon。例如 0.1。默认值为 0.0（不平滑）。
        pos_weight (Tensor, optional): 正样本的权重，用于处理类别不平衡。
    """
    def __init__(self, smoothing=0.0, pos_weight=None):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        # 使用标准的 BCEWithLogitsLoss 作为基础计算单元
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def forward(self, logits, targets):
        """
        前向传播计算损失。

        Args:
            logits (Tensor): 模型的原始输出 [Batch, 1]
            targets (Tensor): 真实的标签 [Batch, 1], 取值为 0 或 1
        """
        # 确保 target 是浮点数
        targets = targets.float()

        # 1. 计算平滑后的软标签 (Soft Targets)
        # 将 0 变为 smoothing, 将 1 变为 1 - smoothing
        # 公式: new_target = target * (1 - smoothing) + smoothing
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing

        # 2. 使用标准的 BCEWithLogitsLoss 计算损失
        # 注意这里传入的是平滑后的标签
        loss = self.bce(logits, smooth_targets)

        # 3. 返回平均损失
        return loss.mean()

# 用于兼容旧代码的辅助函数（现在不需要了，但保留也无妨）
def get_loss_function():
    return nn.BCEWithLogitsLoss()