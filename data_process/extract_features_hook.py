import os
import torch
import torch.optim as optim # Keep this for compatibility, though not used in main
import numpy as np
from tqdm import tqdm
import sys
import argparse # [新增] 导入 argparse

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.sstnet import SSTNet
from utils.dataloader import get_loader


def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="Extract SSTNet features for MIL training.")
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3],
                        help='Which cross-validation fold model to load for feature extraction.')
    args = parser.parse_args()

    FOLD_IDX = args.fold # [修改] 动态设置 Fold Index

    # 1. 配置
    MODEL_PATH = os.path.join(config.PROJECT_ROOT, 'checkpoints', f'best_model_fold{FOLD_IDX}.pth')
    OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, f'mil_features_fold{FOLD_IDX}.npy')

    device = torch.device(config.DEVICE)

    print(f"\n🚀 Starting feature extraction using Fold {FOLD_IDX} model...")
    print(f"💾 Features will be saved to: {OUTPUT_FILE}")

    # 2. 加载模型
    print(f"🔄 Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found! Please run train.py --fold {FOLD_IDX} first.")
        return

    model = SSTNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device) # [优化] 增加 map_location
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. 准备数据
    print("🔄 Preparing DataLoaders (Train + Val)...")
    # 我们需要所有人的数据，所以加载 Fold 0 的 train 和 val 就涵盖了全集
    # [修改] 使用 config.BATCH_SIZE
    loader_train = get_loader('train', fold_idx=0, batch_size=config.BATCH_SIZE)
    loader_val = get_loader('val', fold_idx=0, batch_size=config.BATCH_SIZE)

    # 4. 特征提取主循环
    patient_data = {}

    print("Start extraction...")

    with torch.no_grad():
        for loader in [loader_train, loader_val]:
            # [修改] 解包 6 个变量 (适配 v2.0)
            for local_vis, global_vis, physio, mask, label, subject_ids in tqdm(loader):
                local_vis = local_vis.to(device)
                global_vis = global_vis.to(device).squeeze(1)  # [B, 1, 512] -> [B, 512]
                physio = physio.to(device)
                mask = mask.to(device)

                # [修改] 传入 4 个参数，并开启 return_feats=True
                feats = model(local_vis, global_vis, physio, mask, return_feats=True)

                # 转为 CPU numpy
                feats_np = feats.cpu().numpy()
                labels_np = label.numpy()

                # 归档
                for i, subj_id in enumerate(subject_ids):
                    if subj_id not in patient_data:
                        patient_data[subj_id] = {
                            'features': [],
                            'label': labels_np[i]
                        }
                    patient_data[subj_id]['features'].append(feats_np[i])

    # 5. 整理与保存
    print("📦 Packaging data...")
    final_data = {}
    counts = []

    for subj, data in patient_data.items():
        # [优化] 将特征列表转为 numpy 数组
        feats_matrix = np.array(data['features'], dtype=np.float32)
        label = data['label']

        final_data[subj] = {
            'features': feats_matrix,
            'label': label
        }
        counts.append(len(feats_matrix))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, final_data)

    print(f"✅ Features saved to: {OUTPUT_FILE}")
    print(f"📊 Stats: Total Patients: {len(final_data)}")
    print(f"   Images per Patient: Min={min(counts)}, Max={max(counts)}, Mean={np.mean(counts):.1f}")


if __name__ == "__main__":
    main()