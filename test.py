import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# 导入项目模块
import config
from models.sstnet import SSTNet


# ==========================================
# 1. 定义测试专用的 Dataset
# ==========================================
class TestDataset(Dataset):
    def __init__(self):
        self.max_len = config.MAX_SEQ_LEN

        # 1. 加载测试集视觉特征
        print(f"🔄 Loading TEST visual features from {config.CLIP_TEST_FEATURE_FILE} ...")
        if not os.path.exists(config.CLIP_TEST_FEATURE_FILE):
            raise FileNotFoundError(
                "❌ Test feature file not found! Please run 'generate_clip_features.py --test' first.")

        self.visual_data = np.load(config.CLIP_TEST_FEATURE_FILE, allow_pickle=True).item()

        # 获取所有样本的 key (例如: Test_001_cat)
        self.keys = list(self.visual_data.keys())
        print(f"✅ Test set loaded: {len(self.keys)} sequences found.")

    def __getitem__(self, idx):
        key = self.keys[idx]

        # --- A. 获取视觉特征 ---
        data_pack = self.visual_data[key]
        visual_local = data_pack['local']  # [Seq, 512]
        visual_global = data_pack['global']  # [1, 512]

        # --- B. 获取生理特征 ---
        # 构造对应的 txt 路径
        txt_path = os.path.join(config.TEST_TXT_DIR, f"{key}.txt")

        physio_feat = np.zeros((visual_local.shape[0], config.PHYSIO_DIM), dtype=np.float32)

        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            try:
                # 读取 txt (格式: Index, X, Y, Duration, Pupil)
                df = pd.read_csv(txt_path, header=None)
                raw_data = df.iloc[:, 1:3].values.astype(np.float32)

                # 归一化 (必须与训练时一致!)
                min_vec = np.array([config.SCREEN_X_MIN, config.SCREEN_Y_MIN], dtype=np.float32)
                max_vec = np.array([config.SCREEN_X_MAX, config.SCREEN_Y_MAX], dtype=np.float32)

                raw_data = np.clip(raw_data, min_vec, max_vec)
                norm_0_1 = (raw_data - min_vec) / (max_vec - min_vec + 1e-6)
                physio_feat = norm_0_1 * 2 - 1
            except Exception:
                pass

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

        # 解析 Subject ID (假设文件名格式为 Test_001_ImgName)
        # 如果你的测试集命名不同，这里需要调整
        parts = key.split('_')
        if len(parts) >= 2:
            # 取前两个部分作为 ID，例如 "Test_001"
            subject_id = f"{parts[0]}_{parts[1]}"
        else:
            subject_id = key

        return (
            torch.FloatTensor(padded_local),
            torch.FloatTensor(visual_global),
            torch.FloatTensor(padded_physio),
            torch.FloatTensor(mask),
            subject_id,
            key  # 返回完整文件名以便记录
        )

    def __len__(self):
        return len(self.keys)


# ==========================================
# 2. 主测试流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SSTNet Inference/Test Script")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='test_results.csv', help='Path to save results CSV')
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Inference using: {device}")

    # 1. 加载模型
    print(f"🔄 Loading Model from: {args.ckpt}")
    model = SSTNet().to(device)

    # 加载权重 (处理可能存在的 'model' 键)
    checkpoint = torch.load(args.ckpt, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # 2. 准备数据
    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 推理循环
    results_list = []

    # 用于病人级聚合
    patient_scores = {}

    print("running inference...")
    with torch.no_grad():
        for local_vis, global_vis, physio, mask, subject_ids, keys in tqdm(test_loader):
            local_vis = local_vis.to(device)
            global_vis = global_vis.to(device).squeeze(1)  # [B, 1, 512] -> [B, 512]
            physio = physio.to(device)
            mask = mask.to(device)

            # 前向传播
            logits = model(local_vis, global_vis, physio, mask)

            # 转概率 (Sigmoid)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # 记录结果
            for i in range(len(keys)):
                subj_id = subject_ids[i]
                prob = probs[i]
                fname = keys[i]

                # 存入列表
                results_list.append({
                    "Subject_ID": subj_id,
                    "File_Name": fname,
                    "Probability": prob
                })

                # 聚合逻辑
                if subj_id not in patient_scores:
                    patient_scores[subj_id] = []
                patient_scores[subj_id].append(prob)

    # 4. 计算病人级最终结果 (Mean Voting)
    print("\n📊 Aggregating Patient-Level Results...")
    final_patient_results = []

    for subj_id, scores in patient_scores.items():
        avg_score = np.mean(scores)
        # 假设阈值 0.5，根据需要调整
        pred_label = 1 if avg_score > 0.5 else 0

        final_patient_results.append({
            "Subject_ID": subj_id,
            "Avg_Probability": avg_score,
            "Prediction": pred_label,
            "Sequence_Count": len(scores)
        })

    # 5. 保存文件
    # 保存详细的每张图的结果
    df_detail = pd.DataFrame(results_list)
    detail_csv = args.output.replace('.csv', '_detail.csv')
    df_detail.to_csv(detail_csv, index=False)

    # 保存病人级最终诊断
    df_patient = pd.DataFrame(final_patient_results)
    df_patient.to_csv(args.output, index=False)

    print(f"✅ Done!")
    print(f"   - Detailed results: {detail_csv}")
    print(f"   - Patient diagnosis: {args.output}")


if __name__ == "__main__":
    main()