import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models.temporal_stream import TemporalStream
from models.spatial_stream import NetVLAD


class SSTNet(nn.Module):
    def __init__(self):
        super(SSTNet, self).__init__()

        # 1. 降维层 (512 -> 128)
        self.input_proj = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(0.1)
        )

        # 2. 左臂：时序流 (接收 Global)
        self.temporal_stream = TemporalStream(
            input_dim=config.HIDDEN_DIM,  # 128
            clip_dim=config.INPUT_DIM,  # [新增] 512 (用于Global Proj)
            physio_dim=config.PHYSIO_DIM,
            max_len=config.MAX_SEQ_LEN,
            num_layers=config.TEMP_LAYERS,
            nhead=config.TEMP_HEADS,
            dim_feedforward=config.TEMP_FF_DIM,
            dropout=config.TEMP_DROPOUT
        )

        # 3. 右臂：空间流 (只看 Local)
        self.spatial_stream = NetVLAD(
            dim=config.HIDDEN_DIM,  # 128
            num_clusters=config.SPATIAL_CLUSTERS,
            out_dim=config.SPATIAL_OUT_DIM,  # 128
            alpha=config.SPATIAL_ALPHA
        )

        # 4. 分类头
        fusion_dim = config.HIDDEN_DIM + config.SPATIAL_OUT_DIM
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.CLS_HIDDEN_DIM),
            nn.BatchNorm1d(config.CLS_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(config.CLS_HIDDEN_DIM, 1)
        )

    def forward(self, local_visual, global_visual, physio, mask, return_feats=False):
        # 1. 局部特征降维: [B, 32, 512] -> [B, 32, 128]
        local_low = self.input_proj(local_visual)

        # 2. 时序流 (传入 Global)
        temp_feat = self.temporal_stream(local_low, global_visual, physio, mask)

        # 3. 空间流 (只传 Local)
        spatial_feat = self.spatial_stream(local_low, mask=mask)

        # 4. 融合
        fusion_feat = torch.cat([temp_feat, spatial_feat], dim=1)

        if return_feats:
            return fusion_feat

        logits = self.classifier(fusion_feat)
        return logits


# --- 简单的测试代码 (适配 v2.0 全局感知版) ---
if __name__ == "__main__":
    print("🚀 Testing SSTNet Assembly (v2.0)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SSTNet().to(device)

    # 模拟参数
    B = 2
    L = config.MAX_SEQ_LEN  # 32
    D = config.INPUT_DIM  # 512 (CLIP 原始维度)

    # 1. 构造局部特征 (Local Patches) -> [Batch, 32, 512]
    dummy_local = torch.randn(B, L, D).to(device)

    # 2. [新增] 构造全局特征 (Global Context) -> [Batch, 512]
    # 注意：DataLoader 给出的通常是 [Batch, 512] 或 [Batch, 1, 512]
    # 我们的 forward 会处理它
    dummy_global = torch.randn(B, D).to(device)

    # 3. 构造生理特征 -> [Batch, 32, 4]
    dummy_phy = torch.randn(B, L, 4).to(device)

    # 4. 构造 Mask -> [Batch, 32]
    dummy_mask = torch.ones(B, L).to(device)
    dummy_mask[1, 20:] = 0  # 模拟第二个样本后面是 Padding

    # 执行前向传播
    # 参数顺序: local, global, physio, mask
    output = model(dummy_local, dummy_global, dummy_phy, dummy_mask)

    print(f"Input Local:  {dummy_local.shape}")
    print(f"Input Global: {dummy_global.shape}")
    print(f"Input Physio: {dummy_phy.shape}")
    print(f"Output Shape: {output.shape} (Expect [2, 1])")

    if output.shape == (2, 1):
        print("✅ Assembly Success!")
    else:
        print("❌ Shape mismatch!")