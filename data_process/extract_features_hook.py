import os
import torch
import numpy as np
from tqdm import tqdm
import sys
import os
# 将当前脚本的父目录的父目录（即项目根目录）加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.sstnet import SSTNet
from utils.dataloader import get_loader


def main():
    # 1. 配置
    # 假设我们要用 Fold 2 的模型来提取特征 (因为之前 Fold 2 效果最好)
    FOLD_IDX = 3
    MODEL_PATH = os.path.join(config.PROJECT_ROOT, 'checkpoints', f'best_model_fold{FOLD_IDX}.pth')
    OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, f'mil_features_fold{FOLD_IDX}.npy')

    device = torch.device(config.DEVICE)

    # 2. 加载模型
    print(f"🚀 Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found! Please run train.py --fold {FOLD_IDX} first.")
        return

    model = SSTNet().to(device)
    checkpoint = torch.load(MODEL_PATH)
    # 兼容保存的是 state_dict 还是整个 checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. 准备数据
    # 我们需要提取所有人的数据。
    # 最简单的方法是分别加载 train 和 val，然后合并。
    # 为了确保覆盖 100% 的数据，我们这里加载 Fold 0 的 train 和 val (它们加起来就是全集)
    print("🔄 Preparing DataLoaders...")
    loader_train = get_loader('train', fold_idx=0, batch_size=64)
    loader_val = get_loader('val', fold_idx=0, batch_size=64)

    # 4. 特征提取主循环
    # 字典结构: { '101': {'features': [], 'label': 0}, '102': ... }
    patient_data = {}

    print("Start extraction...")

    with torch.no_grad():
        # 遍历两个 Loader (覆盖所有 160 人)
        for loader in [loader_train, loader_val]:
            for visual, physio, mask, label, subject_ids in tqdm(loader):
                visual = visual.to(device)
                physio = physio.to(device)
                mask = mask.to(device)

                # [关键] 开启 return_feats=True
                # feats: [Batch, 1024]
                feats = model(visual, physio, mask, return_feats=True)

                # 转为 CPU numpy
                feats_np = feats.cpu().numpy()
                labels_np = label.numpy()

                # [核心逻辑] 按 ID 归档
                for i, subj_id in enumerate(subject_ids):
                    # 如果是第一次遇到这个病人，初始化字典
                    if subj_id not in patient_data:
                        patient_data[subj_id] = {
                            'features': [],
                            'label': labels_np[i]  # 记录标签
                        }

                    # 将该图片的特征加入列表
                    patient_data[subj_id]['features'].append(feats_np[i])

    # 5. 整理与保存
    print("📦 Packaging data...")
    final_data = {}

    # 统计一下每个人的图片数量
    counts = []

    for subj, data in patient_data.items():
        # list -> numpy array [N, 1024]
        feats_matrix = np.array(data['features'])
        label = data['label']

        final_data[subj] = {
            'features': feats_matrix,
            'label': label
        }
        counts.append(len(feats_matrix))

    # 保存
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, final_data)

    print(f"✅ Features saved to: {OUTPUT_FILE}")
    print(f"📊 Stats: Total Patients: {len(final_data)}")
    print(f"   Images per Patient: Min={min(counts)}, Max={max(counts)}, Mean={np.mean(counts):.1f}")
    if min(counts) < 100:
        print("ℹ️ Note: Variable sequence lengths detected and handled automatically.")


if __name__ == "__main__":
    main()