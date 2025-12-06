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

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.CLS_HIDDEN_DIM),
            nn.BatchNorm1d(config.CLS_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(config.CLS_HIDDEN_DIM, 1)
        )

    def forward(self, local_visual, global_visual, physio, mask, return_feats=False):
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
        struct_feat = self.structural_stream(local_low, physio)

        # 5. 三流融合
        fusion_feat = torch.cat([temp_feat, spatial_feat, struct_feat], dim=1)

        if return_feats:
            return fusion_feat

        logits = self.classifier(fusion_feat)
        return logits


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
    dummy_physio = torch.rand(B, L, 2).to(device)  # [2, 32, 2] (X,Y 0~1)
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