import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse
import sys

# 引入配置和模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mil_model import RobustGatedAttention
from torch.utils.data import DataLoader, Dataset


# 简单的内存数据集类
class InMemoryMILDataset(Dataset):
    def __init__(self, feature_file, fold_idx, set_name='val'):
        self.data = np.load(feature_file, allow_pickle=True).item()

        # 读取 Excel 进行划分 (复制 train_mil.py 的逻辑)
        import pandas as pd
        excel_path = os.path.join(config.DATASET_DIR, 'Train_Valid.xlsx')
        df = pd.read_excel(excel_path)
        folds = ['Set_0', 'Set_1', 'Set_2', 'Set_3']
        val_col = folds[fold_idx]
        train_cols = [f for f in folds if f != val_col]

        target_ids = []
        if set_name == 'val':
            raw_ids = df[val_col].dropna().values
            target_ids = [str(int(i)).zfill(3) for i in raw_ids]
        else:
            for col in train_cols:
                raw_ids = df[col].dropna().values
                target_ids.extend([str(int(i)).zfill(3) for i in raw_ids])

        self.subjects = [sid for sid in target_ids if sid in self.data]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_id = self.subjects[idx]
        item = self.data[subj_id]
        features = torch.FloatTensor(item['features'])
        label = torch.tensor(item['label'], dtype=torch.float)
        return features, label


def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return sens, spec, ppv, npv


def main():
    device = torch.device(config.DEVICE)
    print("🚀 Starting MIL Industrial Evaluation (Patient Level)...")

    all_labels = []
    all_probs = []

    # 遍历 4 个折
    for fold_idx in range(4):
        print(f"\n🔄 Processing Fold {fold_idx}...")

        # 1. 路径
        feature_file = os.path.join(config.OUTPUT_DIR, f'mil_features_fold{fold_idx}.npy')
        ckpt_path = os.path.join(config.PROJECT_ROOT, 'checkpoints', f'best_mil_model_fold{fold_idx}.pth')

        if not os.path.exists(feature_file) or not os.path.exists(ckpt_path):
            print(f"⚠️ Missing files for Fold {fold_idx}, skipping.")
            continue

        # 2. 数据
        val_set = InMemoryMILDataset(feature_file, fold_idx, set_name='val')
        # 自动探测维度
        sample_feat, _ = val_set[0]
        input_dim = sample_feat.shape[1]

        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

        # 3. 模型
        model = RobustGatedAttention(input_dim=input_dim, bottleneck_dim=32, hidden_dim=128, dropout=0.5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 4. 推理
        with torch.no_grad():
            for features, label in val_loader:
                features = features.to(device)
                logits, _ = model(features)
                prob = torch.sigmoid(logits).item()

                all_probs.append(prob)
                all_labels.append(label.item())

    # === 全局报告 ===
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print(f"🌍 FINAL PATIENT DIAGNOSIS REPORT (N={len(y_true)})")
    print("=" * 60)
    print(f"{'Thres':<6} | {'Sens (漏诊率)':<12} | {'Spec (误诊率)':<12} | {'PPV':<8} | {'NPV':<8}")
    print("-" * 65)

    best_th = 0.5
    best_score = 0

    for th in np.arange(0.3, 0.95, 0.05):
        sens, spec, ppv, npv = calculate_metrics(y_true, y_probs, th)
        print(f"{th:.2f}   | {sens:.4f}       | {spec:.4f}       | {ppv:.4f}   | {npv:.4f}")

        # 寻找甜点：Sens > 0.85 的前提下，Spec 最高
        if sens >= 0.85 and spec > best_score:
            best_score = spec
            best_th = th

    print("-" * 65)
    print(f"💡 Recommended Threshold for Product: {best_th:.2f}")

    # 最终灰区分析
    low_th, high_th = 0.3, 0.7
    uncertain = np.sum((y_probs >= low_th) & (y_probs <= high_th))
    print(f"\n🚦 Gray Zone Analysis ({low_th}-{high_th}):")
    print(f"   Need Manual Review: {uncertain} patients ({uncertain / len(y_true) * 100:.1f}%)")


if __name__ == "__main__":
    main()