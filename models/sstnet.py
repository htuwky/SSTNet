import torch
import torch.nn as nn
import sys
import os

# 导入配置 (这是本项目唯一应该导入 config 的模型文件)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models.temporal_stream import TemporalStream
from models.spatial_stream import NetVLAD


class SSTNet(nn.Module):
    """
    [控制器] SSTNet 主模型
    负责读取 config，实例化子模块，并执行双流前向传播。
    """

    def __init__(self):
        super(SSTNet, self).__init__()

        # === 1. 组装左臂 (时序流) ===
        # 从 config 中提取参数并注入
        self.temporal_stream = TemporalStream(
            input_dim=config.INPUT_DIM,
            physio_dim=config.PHYSIO_DIM,
            max_len=config.MAX_SEQ_LEN,
            num_layers=config.TEMP_LAYERS,
            nhead=config.TEMP_HEADS,
            dim_feedforward=config.TEMP_FF_DIM,
            dropout=config.TEMP_DROPOUT
        )

        # === 2. 组装右臂 (空间流) ===
        # 从 config 中提取参数并注入
        self.spatial_stream = NetVLAD(
            dim=config.INPUT_DIM,
            num_clusters=config.SPATIAL_CLUSTERS,
            out_dim=config.SPATIAL_OUT_DIM,
            alpha=config.SPATIAL_ALPHA
        )

        # === 3. 组装分类头 ===
        fusion_dim = config.INPUT_DIM + config.SPATIAL_OUT_DIM

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.CLS_HIDDEN_DIM),
            nn.BatchNorm1d(config.CLS_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(config.CLS_HIDDEN_DIM, 1)
            # 无 Sigmoid (配合 Focal Loss)
        )

    def forward(self, visual, physio, mask):
        # 分支 A
        temp_feat = self.temporal_stream(visual, physio, mask)

        # 分支 B
        spatial_feat = self.spatial_stream(visual, mask=mask)

        # 融合
        fusion_feat = torch.cat([temp_feat, spatial_feat], dim=1)

        # 诊断
        logits = self.classifier(fusion_feat)

        return logits


# --- 测试代码 ---
if __name__ == "__main__":
    print("🚀 Testing SSTNet Assembly...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SSTNet().to(device)

    # 验证是否能跑通
    B, L, D = 2, config.MAX_SEQ_LEN, config.INPUT_DIM
    dummy_vis = torch.randn(B, L, D).to(device)
    dummy_phy = torch.randn(B, L, 4).to(device)
    dummy_mask = torch.ones(B, L).to(device)

    out = model(dummy_vis, dummy_phy, dummy_mask)
    print(f"Output Shape: {out.shape} (Expect [2, 1])")
    print("✅ Assembly Success!")