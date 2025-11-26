import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import argparse
import sys

# 导入配置和模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保能找到根目录
import config
from models.mil_model import RobustGatedAttention
from utils.misc import fix_seed


# 1. 数据集类 (保持不变)
class MILDataset(Dataset):
    def __init__(self, features_file, fold_idx=0, set_name='train',seed=42):
        print(f"🔄 [{set_name.upper()}] Loading features from: {features_file}")
        try:
            self.data = np.load(features_file, allow_pickle=True).item()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ File not found: {features_file}")

        all_subjects = sorted(list(self.data.keys()))

        # [关键修改] 2. 固定随机种子并打乱
        # 必须保证 train 和 val 看到的打乱顺序是一样的，否则会数据泄漏
        import random
        random.seed(seed)
        random.shuffle(all_subjects)  # <--- 这里打乱！

        N = len(all_subjects)
        fold_size = N // 4
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size

        if set_name == 'val':
            self.subjects = all_subjects[val_start:val_end]
        else:
            self.subjects = all_subjects[:val_start] + all_subjects[val_end:]

        print(f"✅ Loaded {len(self.subjects)} patients.")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_id = self.subjects[idx]
        item = self.data[subj_id]
        features = torch.FloatTensor(item['features'])
        label = torch.tensor(item['label'], dtype=torch.float)
        return features, label


# 2. 训练函数 (保持不变)
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for features, label in loader:
        features, label = features.to(device), label.to(device)
        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits.squeeze(1), label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


# 3. 验证函数 (保持不变)
def validate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features, label in loader:
            features = features.to(device)
            logits, _ = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(label.numpy())
    try:
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    except ValueError:
        auc = 0.5
        acc = 0.5
    return auc, acc


# 4. 主函数 (核心修改)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=2, help='Which fold features to use')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 路径构建
    feature_file = os.path.join(config.OUTPUT_DIR, f'mil_features_fold{args.fold}.npy')
    if not os.path.exists(feature_file):
        print(f"❌ Error: Feature file not found at {feature_file}")
        return

    # 加载数据
    train_set = MILDataset(feature_file, fold_idx=0, set_name='train', seed=args.seed)
    val_set = MILDataset(feature_file, fold_idx=0, set_name='val', seed=args.seed)

    # [关键修改] 自动探测维度
    sample_feat, _ = train_set[0]
    detected_dim = sample_feat.shape[1]
    print(f"🚀 Start MIL Training (Robust Gated Attention)")
    print(f"   Feature Source: Fold {args.fold}")
    print(f"   Detected Dimension: {detected_dim}")  # 应该会打印 2560
    print(f"   Params: LR={args.lr}, Dropout={args.dropout}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 构建模型 (使用检测到的维度)
    model = RobustGatedAttention(input_dim=detected_dim, hidden_dim=256, dropout=args.dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        auc, acc = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1:02d}: Loss={train_loss:.4f}, Val AUC={auc:.4f}, Acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_mil_model.pth")
            print("  --> 💾 New Best Saved!")

    print("=" * 40)
    print(f"🏆 Final Best Patient-Level AUC: {best_auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()