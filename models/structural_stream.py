import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torch.nn.functional as F
# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class StructuralStream(nn.Module):
    """
    [纯原生版] 结构流 GNN
    完全移除 PyG 依赖，使用原生 PyTorch 实现 KNN 和 EdgeConv。
    功能与 PyG 版完全一致，但兼容性无敌。
    """

    def __init__(self):
        super().__init__()

        self.k = config.GNN_K
        dim = config.HIDDEN_DIM
        dropout = config.GNN_DROPOUT

        # EdgeConv 的核心 MLP: h(x_i, x_j - x_i)
        # 输入维度: 2*dim (中心点 + 相对特征)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            # 使用 LayerNorm 替代 BatchNorm，对序列数据更友好且无需转置
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

        # 最终输出映射
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def get_knn_indices(self, coords, k):
        """
        计算 k 近邻索引 (基于欧氏距离)
        coords: [B, N, 2]
        Returns: [B, N, k]
        """
        # 计算两两距离矩阵 (利用公式 (a-b)^2 = a^2 + b^2 - 2ab)
        # inner: [B, N, N]
        inner = -2 * torch.matmul(coords, coords.transpose(2, 1))
        # xx: [B, N, 1]
        xx = torch.sum(coords ** 2, dim=2, keepdim=True)
        # pairwise_distance: [B, N, N]
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        # 取最近的 k 个 (注意：topk 取的是最大值，所以我们用了负距离)
        # idx: [B, N, k]
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k, idx):
        """
        根据索引聚合邻居特征 (类似 PyG 的 gather)
        x: [B, N, C]
        idx: [B, N, k]
        """
        batch_size, num_points, channels = x.size()
        device = x.device

        # 1. 构造中心点特征 [B, N, k, C]
        x_central = x.view(batch_size, num_points, 1, channels).repeat(1, 1, k, 1)

        # 2. 构造邻居索引 (处理 Batch 维度)
        # idx 是每个样本内部的索引 (0~N-1)，我们需要用 gather 抓取
        # 扩展 idx 到 [B, N, k, C] 以便 gather
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, channels)

        # 3. 抓取邻居特征
        # x: [B, N, C] -> [B, N, 1, C] 用于 gather 可能会麻烦
        # 更简单的方法是展平 Batch 维度进行索引，或者使用 torch.gather

        # 方案：使用 gather
        # x_neighbors: [B, N, k, C]
        # 我们需要在 dim=1 (N维度) 上 gather
        # 为了方便，先把 x 扩展一下: [B, N, C]
        # 但 gather 需要 index 和 input 维度一致。
        # 最稳妥的纯 PyTorch 写法是利用 batch_index 偏移

        # [B, N*k]
        idx_flat = idx.view(batch_size, -1)

        # 生成 batch 偏移: [0, 0...], [N, N...], [2N, 2N...]
        batch_offset = torch.arange(batch_size, device=device) * num_points
        batch_offset = batch_offset.view(-1, 1)

        # 全局索引: [B, N*k]
        global_idx = idx_flat + batch_offset
        global_idx = global_idx.view(-1)  # [B*N*k]

        # 展平 x: [B*N, C]
        x_flat = x.view(batch_size * num_points, channels)

        # 索引提取: [B*N*k, C]
        feature_flat = x_flat[global_idx]

        # 恢复形状: [B, N, k, C]
        x_neighbors = feature_flat.view(batch_size, num_points, k, channels)

        # 4. 拼接: [x_central, x_neighbors - x_central]
        # [B, N, k, 2*C]
        feature = torch.cat((x_central, x_neighbors - x_central), dim=3)

        return feature

    # models/structural_stream.py

    # [修改] 增加 mask 参数
    def forward(self, x, coords, mask):
        """
        Args:
            x:      [Batch, Seq, Dim]
            coords: [Batch, Seq, 2]
            mask:   [Batch, Seq] (1=真实点, 0=填充点)
        """
        B, N, C = x.shape

        # 1. 动态建图 (KNN)
        x_norm = F.normalize(x, p=2, dim=-1) # [B, N, 128]
        knn_idx = self.get_knn_indices(x_norm, self.k)

        # 2. 构造图特征
        edge_feat = self.get_graph_feature(x, self.k, knn_idx)

        # 3. 消息传递
        edge_feat = self.mlp(edge_feat)

        # 4. 局部聚合 (Max Pooling over neighbors)
        # [B, N, C]
        local_graph_feat = edge_feat.max(dim=2)[0]

        # ================= [新增核心] Masking 机制 =================
        # 防止 Max Pooling 选中填充点产生的噪音特征

        # mask: [B, N] -> [B, N, 1]
        mask_expanded = mask.unsqueeze(-1)

        # 将填充位置的特征设为负无穷 (-1e9)
        # 这样在做下面的 max(dim=1) 时，它们绝对不会被选中
        local_graph_feat = local_graph_feat.masked_fill(mask_expanded == 0, -1e9)
        # =========================================================

        # 5. 全局池化 (Readout)
        # [B, C]
        global_graph_feat = local_graph_feat.max(dim=1)[0]

        # 6. 后处理
        out = self.out_proj(global_graph_feat)
        out = self.norm(out)

        return out