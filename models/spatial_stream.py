import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """
    [纯净组件] NetVLAD 空间流模块
    负责捕捉特征的空间分布统计信息。
    不依赖外部配置文件。
    """

    def __init__(self, dim=128, num_clusters=8, alpha=100.0, out_dim=128):
        """
        Args:
            dim: 输入特征维度
            num_clusters: 聚类中心数量 K
            alpha: 软分配系数
            out_dim: 输出投影维度
        """
        super(NetVLAD, self).__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.alpha = alpha

        # 1. 软分配控制器
        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=False)

        # 2. 聚类中心
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        # 3. 输出投影
        self.proj = nn.Linear(num_clusters * dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.centroids, std=0.01)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Seq, Dim]
            mask: [Batch, Seq]
        """
        B, N, C = x.shape
        x_perm = x.permute(0, 2, 1)

        # 计算分配得分
        assignment = self.conv(x_perm)

        # Mask 处理 (防止填充点干扰聚类)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            assignment = assignment.masked_fill(mask_expanded == 0, -1e9)

        # Softmax 归一化
        a = F.softmax(self.alpha * assignment, dim=1)

        # VLAD 计算
        sum_x = torch.bmm(a, x)
        probs_sum = torch.sum(a, dim=-1, keepdim=True)
        cluster_weighted = probs_sum * self.centroids.unsqueeze(0)
        vlad = sum_x - cluster_weighted

        # 归一化与投影
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(B, -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        out = self.proj(vlad)
        out = self.norm(out)

        return out