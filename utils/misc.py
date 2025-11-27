import torch
import numpy as np
import random
import os

# 杂项工具（固定随机种子、保存检查点）。
def fix_seed(seed=42):
    """
    固定所有随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 CuDNN 的确定性 (会牺牲一点点速度，但保证结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 Random Seed fixed to {seed}")


def save_checkpoint(model, optimizer, epoch, metric, filename):
    """保存模型权重"""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': metric
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"💾 Model saved to {filename}")