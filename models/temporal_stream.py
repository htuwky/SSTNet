import torch
import torch.nn as nn


class TemporalStream(nn.Module):
    """
    [纯净组件] 时序流模块
    负责处理多模态融合与时序逻辑推理。
    不依赖外部配置文件，参数全部由 __init__ 传入。
    """

    def __init__(self,
                 input_dim=512,
                 physio_dim=4,
                 max_len=32,
                 num_layers=2,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.2):
        """
        Args:
            input_dim: 视觉特征维度 (例如 512)
            physio_dim: 生理特征维度 (例如 4)
            max_len: 最大序列长度 (例如 32)
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

        # 2. 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, input_dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 最终归一化
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, visual, physio, mask):
        """
        Args:
            visual: [Batch, Seq, Dim]
            physio: [Batch, Seq, Physio_Dim]
            mask:   [Batch, Seq] (1=Real, 0=Pad)
        """
        B, L, D = visual.shape

        # --- A. 融合 ---
        physio_emb = self.physio_proj(physio)
        # 广播位置编码: [1, Max, Dim] -> [B, L, Dim]
        x = visual + physio_emb + self.pos_embed[:, :L, :]

        # --- B. Transformer ---
        # Mask 取反 (True=Padding/Ignore)
        key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # --- C. Masked Mean Pooling ---
        # 扩展 mask: [B, L, 1]
        mask_expanded = mask.unsqueeze(-1)

        # 求和 (利用 mask 清除 padding 的噪音)
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)

        # 求真实长度
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除以0

        pooled_feat = sum_embeddings / sum_mask

        return self.norm(pooled_feat)