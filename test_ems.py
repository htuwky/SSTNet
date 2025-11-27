import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# 导入项目配置和模型
import config
from models.sstnet import SSTNet


# ==========================================
# 1. 定义测试专用 Dataset
# ==========================================
class EMSTestDataset(Dataset):
    def __init__(self):
        self.max_len = config.MAX_SEQ_LEN
        self.feature_path = config.CLIP_TEST_FEATURE_FILE
        self.txt_root = config.TEST_TXT_DIR

        print(f"🔄 Loading EMS Test features from: {self.feature_path}")
        if not os.path.exists(self.feature_path):
            raise FileNotFoundError(
                f"❌ Feature file not found: {self.feature_path}\nPlease run 'python data_process/generate_clip_features.py --test' first.")

        self.visual_data = np.load(self.feature_path, allow_pickle=True).item()

        # 获取所有序列 Key
        self.keys = list(self.visual_data.keys())
        print(f"✅ Loaded {len(self.keys)} sequences.")

        # 简单统计一下包含多少个受试者
        subjects = set()
        for k in self.keys:
            # 假设文件名格式: Test_001_ImageName...
            parts = k.split('_')
            if len(parts) >= 2:
                subjects.add(f"{parts[0]}_{parts[1]}")
        print(f"👥 Unique Subjects identified: {len(subjects)} (Target is ~48)")

    def __getitem__(self, idx):
        key = self.keys[idx]

        # --- A. 获取视觉特征 ---
        data_pack = self.visual_data[key]
        visual_local = data_pack['local']  # [Seq, 512]
        visual_global = data_pack['global']  # [1, 512] 或者 [512]

        # --- B. 获取生理特征 ---
        # 构造 txt 路径
        txt_path = os.path.join(self.txt_root, f"{key}.txt")

        physio_feat = np.zeros((visual_local.shape[0], config.PHYSIO_DIM), dtype=np.float32)

        # 尝试读取并归一化生理数据
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            try:
                # 读取 txt (Index, X, Y, Duration, Pupil)
                df = pd.read_csv(txt_path, header=None)
                raw_data = df.iloc[:, 1:5].values.astype(np.float32)

                # 归一化 (使用 config 中的参数)
                min_vec = np.array([config.SCREEN_X_MIN, config.SCREEN_Y_MIN, config.DUR_MIN, config.PUPIL_MIN],
                                   dtype=np.float32)
                max_vec = np.array([config.SCREEN_X_MAX, config.SCREEN_Y_MAX, config.DUR_MAX, config.PUPIL_MAX],
                                   dtype=np.float32)

                raw_data = np.clip(raw_data, min_vec, max_vec)
                norm_0_1 = (raw_data - min_vec) / (max_vec - min_vec + 1e-6)
                physio_feat = norm_0_1 * 2 - 1  # 映射到 [-1, 1]
            except Exception:
                pass  # 出错保持全0

        # --- C. Padding & Mask ---
        seq_len = visual_local.shape[0]
        target_len = self.max_len

        padded_local = np.zeros((target_len, config.INPUT_DIM), dtype=np.float32)
        padded_physio = np.zeros((target_len, config.PHYSIO_DIM), dtype=np.float32)
        mask = np.zeros(target_len, dtype=np.float32)

        valid_len = min(seq_len, target_len)

        padded_local[:valid_len] = visual_local[:valid_len]

        fill_len = min(valid_len, physio_feat.shape[0])
        padded_physio[:fill_len] = physio_feat[:fill_len]
        mask[:fill_len] = 1.0

        # 解析 Subject ID (格式: Test_001)
        parts = key.split('_')
        if len(parts) >= 2:
            subject_id = f"{parts[0]}_{parts[1]}"
        else:
            subject_id = key

        return (
            torch.FloatTensor(padded_local),
            torch.FloatTensor(visual_global),
            torch.FloatTensor(padded_physio),
            torch.FloatTensor(mask),
            subject_id
        )

    def __len__(self):
        return len(self.keys)


# ==========================================
# 2. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate prob_test.txt for EMS dataset")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarization')
    args = parser.parse_args()

    # 输出文件路径
    output_file = 'prob_test.txt'

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"\n🚀 Loading Model from: {args.ckpt}")
    model = SSTNet().to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    # 兼容不同的保存格式
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. 准备数据
    test_dataset = EMSTestDataset()
    # Batch Size 设大一点可以加快推理速度
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 推理与聚合
    patient_scores = {}  # { 'Test_001': [0.1, 0.2, ...], ... }

    print(f"🔮 Running Inference on {device}...")

    with torch.no_grad():
        for local_vis, global_vis, physio, mask, subject_ids in tqdm(test_loader):
            local_vis = local_vis.to(device)
            # 处理 global 维度: [B, 1, 512] -> [B, 512]
            if global_vis.dim() == 3:
                global_vis = global_vis.squeeze(1)
            global_vis = global_vis.to(device)

            physio = physio.to(device)
            mask = mask.to(device)

            # 前向传播
            logits = model(local_vis, global_vis, physio, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # 按病人聚合
            for i, subj_id in enumerate(subject_ids):
                if subj_id not in patient_scores:
                    patient_scores[subj_id] = []
                patient_scores[subj_id].append(probs[i])

    # 4. 生成结果文件
    print(f"\n💾 Generating {output_file} ...")

    # 获取排序后的 Subject ID 列表 (确保输出顺序整洁)
    sorted_subjects = sorted(patient_scores.keys())

    lines_written = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for subj_id in sorted_subjects:
            scores = patient_scores[subj_id]

            # 计算平均概率 (Mean Voting)
            avg_prob = np.mean(scores)

            # 二值化
            label = 1 if avg_prob > args.threshold else 0

            # 格式: Test_028,0.798709,1
            line = f"{subj_id},{avg_prob:.6f},{label}"
            f.write(line + '\n')
            lines_written += 1

    print(f"✅ Done! Written {lines_written} subjects to {output_file}.")
    print(f"   (Threshold used: {args.threshold})")


if __name__ == "__main__":
    main()