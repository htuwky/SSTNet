import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustGatedAttention(nn.Module):
    """
    [MIL Aggregator] Robust Gated Attention Network
    专门用于处理多示例学习 (MIL) 的聚合任务。
    特点：
    1. [新增] Bottleneck 降维：在注意力机制前将高维特征压缩，防止过拟合。
    2. Gating 机制 (tanh * sigmoid)：精准识别关键样本(Key Instance)。
    3. 强正则化 (LayerNorm + Dropout)：防止在小样本(160人)上过拟合。
    """

    def __init__(self, input_dim=256, bottleneck_dim=64, hidden_dim=128, dropout=0.5):
        """
        Args:
            input_dim: 输入特征维度 (SSTNet融合后的维度，Detect出来是 256)
            bottleneck_dim: [新增] 降维后的维度 (建议 64 或 32，用于减少参数)
            hidden_dim: 内部注意力层的宽度 (建议 128)
            dropout: 丢弃率 (建议 0.5)
        """
        super().__init__()

        # --- [新增] 0. 降维层 (Bottleneck) ---
        # 将输入的 256 维压缩到 bottleneck_dim (如 64)
        # 这一步大幅减少了后续注意力层的参数量
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- 1. 特征预处理 (针对降维后的特征) ---
        # LayerNorm: 将不同病人的特征拉到同一水平线，加速收敛
        # [修改] 输入维度变为 bottleneck_dim
        self.layer_norm = nn.LayerNorm(bottleneck_dim)

        # --- 2. 门控注意力核心 (Gated Attention Mechanism) ---
        # [修改] 输入维度变为 bottleneck_dim
        # 路径 V (Content): 提取特征的内容信息
        self.attention_V = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Tanh()
        )
        # 路径 U (Gate): 提取特征的重要性/门控信息
        self.attention_U = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Sigmoid()
        )

        # --- 3. 权重计算 ---
        # Attention Dropout: 防止模型过度依赖某几张特定的图
        self.attn_dropout = nn.Dropout(dropout)
        # 将门控特征映射为一个标量分数
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # --- 4. 病人级分类器 (Classifier) ---
        # 输入是聚合后的 1 个特征向量，输出患病概率 Logits
        # [修改] 输入维度变为 bottleneck_dim
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # 强力防止过拟合
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [Batch=1, Bag_Size, Input_Dim(256)]
               (注意：Bag_Size 是变长的)
        Returns:
            logits:  [1, 1] (预测分数)
            weights: [1, Bag_Size] (每张图的注意力权重，用于可视化)
        """

        # [新增] 0. 先降维
        # x shape: [1, N, 256] -> [1, N, 64]
        x = self.input_proj(x)

        # 1. 预处理 (后续操作基于降维后的特征)
        x = self.layer_norm(x)
        # x_drop = self.feat_dropout(x) # input_proj 里已经有 dropout 了，这里可以去掉

        # 2. 计算 Attention Score
        a_v = self.attention_V(x)  # [1, N, Hidden]
        a_u = self.attention_U(x)  # [1, N, Hidden]

        # Gating: 捕捉非线性关系
        gated = a_v * a_u
        gated = self.attn_dropout(gated)

        # 计算未归一化分数
        scores = self.attention_weights(gated)  # [1, N, 1]

        # Transpose for Softmax: [1, 1, N]
        scores = scores.transpose(1, 2)

        # Softmax 归一化 (dim=2 是图片数量维度)
        # 得到每张图的权重，和为 1
        weights = torch.softmax(scores, dim=2)  # [1, 1, N]

        # 3. 聚合 (Aggregation)
        # 加权求和: weights * x
        # [1, 1, N] @ [1, N, Bottleneck_Dim] -> [1, 1, Bottleneck_Dim]
        patient_feat = torch.bmm(weights, x)

        # Squeeze: [1, Bottleneck_Dim]
        patient_feat = patient_feat.squeeze(1)

        # 4. 分类
        logits = self.classifier(patient_feat)

        return logits, weights