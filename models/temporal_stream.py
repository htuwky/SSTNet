import torch
import torch.nn as nn


class TemporalStream(nn.Module):
    """
    [纯净组件] 时序流模块 (v2.0 Global Aware)
    负责处理多模态融合与时序逻辑推理。
    支持融合 全局背景特征 + 局部视觉特征 + 生理特征。
    """

    def __init__(self,
                 input_dim=128,  # 保持你要求的默认值
                 clip_dim=512,  # [新增] CLIP 原始维度 (用于全局特征投影)
                 physio_dim=4,
                 max_len=32,
                 num_layers=2,
                 nhead=4,  # 保持你要求的默认值
                 dim_feedforward=512,  # 保持你要求的默认值
                 dropout=0.5):  # 保持你要求的默认值
        """
        Args:
            input_dim: 输入/内部计算维度
            clip_dim: 全局特征的原始维度 (通常为512)
            physio_dim: 生理特征维度
            max_len: 最大序列长度
            num_layers: Transformer 层数
            nhead: 注意力头数
            dim_feedforward: 前馈层维度
            dropout: Dropout 比率
        """
        super().__init__()

        self.input_dim = input_dim

        # 1. 生理特征投影层 (低维 -> 高维)
        self.physio_proj = nn.Sequential(
            nn.Linear(physio_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim)
        )

        # 2. [新增] 全局特征投影层
        # 将 CLIP 的全局特征 (512) 映射到内部维度 (input_dim)
        # 即使 input_dim 也是 512，这一层也有助于特征空间的对齐
        self.global_proj = nn.Sequential(
            nn.Linear(clip_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim)
        )

        # 3. 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, input_dim) * 0.02)

        # 4. [新增] 融合归一化层 (Fusion Norm)
        # 用于处理相加后的数值分布，保证 Transformer 输入稳定
        self.fusion_norm = nn.LayerNorm(input_dim)

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 推荐开启 Pre-Norm，训练更稳
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 6. 最终输出归一化
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, local_visual, global_visual, physio, mask):
        """
        Args:
            local_visual:  [Batch, Seq, Dim] (局部 Patch 特征)
            global_visual: [Batch, Clip_Dim] (全局 Context 特征) <--- 新增
            physio:        [Batch, Seq, Physio_Dim]
            mask:          [Batch, Seq] (1=Real, 0=Pad)
        """
        B, L, D = local_visual.shape

        # --- A. 投影对齐 ---
        # 1. 生理投影: [B, L, 4] -> [B, L, Dim]
        physio_emb = self.physio_proj(physio)

        # 2. 全局投影: [B, Clip_Dim] -> [B, Dim] -> [B, 1, Dim]
        # unsqueeze(1) 是为了后续广播加到每一帧上
        global_emb = self.global_proj(global_visual).unsqueeze(1)

        # --- B. 广播融合 (Broadcast Addition) ---
        # Local + Physio + Global + Pos
        # 这里的 global_emb 会自动广播，加到序列的每一个时间步
        x = local_visual + physio_emb + global_emb + self.pos_embed[:, :L, :]

        # --- C. 融合后归一化 ---
        # 消除多路相加带来的方差变大问题
        x = self.fusion_norm(x)

        # --- D. Transformer 计算 ---
        # Mask 取反 (True=Padding/Ignore)
        key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # --- E. Masked Mean Pooling ---
        # 扩展 mask: [B, L, 1]
        mask_expanded = mask.unsqueeze(-1)

        # 求和 (利用 mask 清除 padding 的噪音)
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)

        # 求真实长度
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除以0

        pooled_feat = sum_embeddings / sum_mask

        return self.norm(pooled_feat)