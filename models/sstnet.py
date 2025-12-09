import torch
import torch.nn as nn
import sys
import os

# 将项目根目录加入路径，确保能导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models.temporal_stream import TemporalStream
from models.spatial_stream import NetVLAD
# [新增] 导入结构流 (GNN)
from models.structural_stream import StructuralStream


class SSTNet(nn.Module):
    """
    SSTNet v3.0: Tri-stream Architecture (Spatio-Temporal-Structural)
    融合了三种视角的特征：
    1. Temporal Stream (Transformer): 捕捉时序逻辑与多模态上下文。
    2. Spatial Stream (NetVLAD): 捕捉全局内容统计分布。
    3. Structural Stream (GNN): 捕捉局部注视点的几何拓扑结构。
    """

    def __init__(self):
        super(SSTNet, self).__init__()

        # 1. 局部特征降维层 (CLIP 512 -> 128)
        self.input_proj = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(0.1)
        )

        # 2. 左路：时序流 (Transformer) - The "Time"
        self.temporal_stream = TemporalStream(
            input_dim=config.HIDDEN_DIM,
            clip_dim=config.INPUT_DIM,
            physio_dim=config.PHYSIO_DIM,  # 2 (x, y)
            max_len=config.MAX_SEQ_LEN,
            num_layers=config.TEMP_LAYERS,
            nhead=config.TEMP_HEADS,
            dim_feedforward=config.TEMP_FF_DIM,
            dropout=config.TEMP_DROPOUT
        )

        # 3. 右路：空间流 (NetVLAD) - The "Content"
        self.spatial_stream = NetVLAD(
            dim=config.HIDDEN_DIM,
            num_clusters=config.SPATIAL_CLUSTERS,
            out_dim=config.SPATIAL_OUT_DIM,
            alpha=config.SPATIAL_ALPHA
        )

        # [新增] 4. 中路：结构流 (GNN) - The "Structure"
        # 使用 PyG 实现的动态图卷积
        self.structural_stream = StructuralStream()

        # 5. 分类头 (Classifier)
        # 融合维度 = 128 (Time) + 128 (Space) + 128 (Structure) = 384
        fusion_dim = config.HIDDEN_DIM * 3
        self.stream_attention = StreamAttention(total_dim=config.HIDDEN_DIM * 3, num_streams=3)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.CLS_HIDDEN_DIM),
            nn.BatchNorm1d(config.CLS_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(config.CLS_HIDDEN_DIM, 1)
        )
        # [新增] 6. 对比学习投影头 (Projection Head)
        # 将 384 维特征映射到 128 维用于计算对比 Loss
        self.projector = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 3, config.HIDDEN_DIM * 3),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM * 3, 128)  # 压缩到 128 维
        )

    def forward(self, local_visual, global_visual, physio, mask, return_feats=False, return_proj=False):
        """
        Args:
            local_visual:  [Batch, Seq, 512]
            global_visual: [Batch, 512]
            physio:        [Batch, Seq, 2] (x, y)
            mask:          [Batch, Seq]
        """
        # 1. 局部特征降维 [B, N, 512] -> [B, N, 128]
        local_low = self.input_proj(local_visual)

        # 2. 时序流 (Temporal)
        temp_feat = self.temporal_stream(local_low, global_visual, physio, mask)

        # 3. 空间流 (Spatial)
        spatial_feat = self.spatial_stream(local_low, mask=mask)

        # [新增] 4. 结构流 (Structural)
        # 注意：显式传入 physio (x,y) 用于动态建图 (KNN)
        # struct_feat = self.structural_stream(local_low, physio)
        struct_feat = self.structural_stream(local_low, physio, mask)
        # 5. 三流融合
        # fusion_feat = torch.cat([temp_feat, spatial_feat, struct_feat], dim=1)
        stream_list = [temp_feat, spatial_feat, struct_feat]
        fusion_feat, weights = self.stream_attention(stream_list)
        # [修改] 各种返回模式
        if return_feats:
            return fusion_feat  # 供 MIL 提取特征用

        logits = self.classifier(fusion_feat)

        # [新增] 如果训练时开启对比学习，返回 (logits, projections)
        if return_proj:
            # 投影并归一化 (SupCon 必须归一化)
            proj = self.projector(fusion_feat)
            proj = torch.nn.functional.normalize(proj, dim=1)
            return logits, proj

        return logits


class StreamAttention(nn.Module):
    """
    [新增组件] 流注意力模块
    根据输入特征动态计算三个流的权重 (w1, w2, w3)
    """

    def __init__(self, total_dim=384, num_streams=3):
        super().__init__()
        # 一个非常小的 MLP: 384 -> 64 -> 3
        self.attn_net = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),  # 降维减少参数
            nn.ReLU(),
            nn.Linear(total_dim // 4, num_streams),
            nn.Softmax(dim=1)  # 保证权重之和为 1
        )

    def forward(self, x_list):
        """
        Args:
            x_list: [feat1, feat2, feat3] 列表
        """
        # 1. 先拼接拿到全量特征
        cat_feat = torch.cat(x_list, dim=1)

        # 2. 计算权重 [Batch, 3]
        weights = self.attn_net(cat_feat)

        # 3. 加权 (注意: 保持维度独立，不是相加，而是加权拼接)
        # 我们希望保留 384 维，只是让某些流变强/变弱
        weighted_list = []
        for i, feat in enumerate(x_list):
            # weights[:, i] 形状是 [B], 需要 unsqueeze 成 [B, 1]
            w = weights[:, i].unsqueeze(1)
            weighted_list.append(feat * w)

        # 4. 再次拼接作为输出
        return torch.cat(weighted_list, dim=1), weights
# ==========================================
# 简单的自检代码 (运行 python models/sstnet.py)
# ==========================================
if __name__ == "__main__":
    print("🚀 Testing SSTNet v3.0 (Tri-stream)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 尝试初始化模型
    try:
        model = SSTNet().to(device)
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        exit()

    # 模拟输入数据
    B = 2
    L = config.MAX_SEQ_LEN
    D_in = config.INPUT_DIM

    dummy_local = torch.randn(B, L, D_in).to(device)  # [2, 32, 512]
    dummy_global = torch.randn(B, D_in).to(device)  # [2, 512]
    dummy_physio = torch.rand(B, L, config.PHYSIO_DIM).to(device)  # [2, 32, 2] (X,Y 0~1)
    dummy_mask = torch.ones(B, L).to(device)  # [2, 32]

    # 执行前向传播
    try:
        output = model(dummy_local, dummy_global, dummy_physio, dummy_mask)
        print(f"✅ Forward pass successful!")
        print(f"   Input Shapes: Local={dummy_local.shape}, Global={dummy_global.shape}, Physio={dummy_physio.shape}")
        print(f"   Output Shape: {output.shape} (Expected: [2, 1])")

        # 检查融合维度是否正确 (简单反推)
        # fusion_dim 应该是 128*3=384。可以在这里打印模型参数量确认。
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total Parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()