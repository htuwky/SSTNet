import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.signal import savgol_filter  # [新增] 导入平滑工具

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SSTDataset(Dataset):
    def __init__(self, set_name, fold_idx=0):
        self.set_name = set_name
        self.max_len = config.MAX_SEQ_LEN

        # 加载特征
        print(f"🔄 [{set_name.upper()}] Loading visual features from {config.CLIP_TRAIN_FEATURE_FILE} ...")
        try:
            self.visual_data = np.load(config.CLIP_TRAIN_FEATURE_FILE, allow_pickle=True).item()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Feature file not found! Run generate_clip_features.py --train first.")

        self.samples = self._split_dataset(fold_idx)
        print(f"✅ {set_name.upper()} set loaded: {len(self.samples)} samples.")

    def _split_dataset(self, fold_idx):
        excel_path = os.path.join(config.DATASET_DIR, 'Train_Valid.xlsx')
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"❌ Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path)
        folds = ['Set_0', 'Set_1', 'Set_2', 'Set_3']
        val_col = folds[fold_idx]
        train_cols = [f for f in folds if f != val_col]

        target_ids = []
        if self.set_name == 'val':
            raw_ids = df[val_col].dropna().values
            target_ids = [str(int(i)).zfill(3) for i in raw_ids]
        else:
            for col in train_cols:
                raw_ids = df[col].dropna().values
                target_ids.extend([str(int(i)).zfill(3) for i in raw_ids])

        sample_list = []
        txt_folder = config.TRAIN_TXT_DIR
        all_files = os.listdir(txt_folder)

        for f in all_files:
            if not f.endswith('.txt'): continue
            filename_no_ext = f.split('.txt')[0]
            try:
                subj_id, _ = filename_no_ext.split('_', 1)
            except ValueError:
                continue

            if subj_id in target_ids:
                label = 0 if int(subj_id) < 200 else 1
                if filename_no_ext in self.visual_data:
                    sample_list.append({
                        'key': filename_no_ext,
                        'txt_path': os.path.join(txt_folder, f),
                        'label': label
                    })
        return sample_list

    def __getitem__(self, idx):
        item = self.samples[idx]
        key = item['key']
        txt_path = item['txt_path']
        label = item['label']

        # --- A. 获取双流视觉特征 ---
        data_pack = self.visual_data[key]
        visual_local = data_pack['local']  # [Seq, 512]
        visual_global = data_pack['global']  # [1, 512]

        # --- B. 获取生理特征 ---
        curr_len = visual_local.shape[0]

        # 先初始化为 2维 (x,y)
        physio_base = np.zeros((curr_len, 2), dtype=np.float32)
        valid_p_len = 0  # 记录有效长度

        try:
            df = pd.read_csv(txt_path, header=None)
            raw_data = df.iloc[:, 1:3].values.astype(np.float32)  # 只读 X, Y

            # 归一化
            min_vec = np.array([config.SCREEN_X_MIN, config.SCREEN_Y_MIN], dtype=np.float32)
            max_vec = np.array([config.SCREEN_X_MAX, config.SCREEN_Y_MAX], dtype=np.float32)

            raw_data = np.clip(raw_data, min_vec, max_vec)
            norm_0_1 = (raw_data - min_vec) / (max_vec - min_vec + 1e-6)

            # 赋值
            valid_p_len = min(len(norm_0_1), curr_len)
            physio_base[:valid_p_len] = norm_0_1[:valid_p_len] * 2 - 1

        except Exception:
            pass

        # ================= [新增] 坐标平滑 (Smoothing) =================
        # 去除高频抖动，让 diff 计算出的速度更真实
        try:
            if valid_p_len > 5:
                # 只平滑有效区域，防止边缘效应
                physio_base[:valid_p_len, 0] = savgol_filter(physio_base[:valid_p_len, 0], 5, 2)
                physio_base[:valid_p_len, 1] = savgol_filter(physio_base[:valid_p_len, 1], 5, 2)
        except Exception:
            pass
        # ==============================================================

        # 1. 计算速度差分特征 (Delta X, Delta Y)
        # 现在的 diff 是基于“平滑后”的坐标算的
        diff = np.zeros_like(physio_base)
        diff[1:] = physio_base[1:] - physio_base[:-1]

        # 2. 拼接成 4维特征 (X, Y, dX, dY)
        physio_feat_4d = np.concatenate([physio_base, diff], axis=-1)

        # --- 数据增强 (仅训练集) ---
        if self.set_name == 'train':
            # 1. 特征加噪
            noise_level = 0.015
            noise = np.random.normal(0, noise_level, visual_local.shape).astype(np.float32)
            visual_local = visual_local + noise

            # 2. 时序随机丢弃
            drop_prob = 0.15
            seq_len_origin = visual_local.shape[0]
            drop_mask = (np.random.rand(seq_len_origin) > drop_prob).astype(np.float32)
            drop_mask = drop_mask[:, np.newaxis]

            visual_local = visual_local * drop_mask
            physio_feat_4d = physio_feat_4d * drop_mask

        # --- C. 统一长度 (Padding) ---
        target_len = self.max_len
        valid_len = min(curr_len, target_len)

        padded_local = np.zeros((target_len, config.INPUT_DIM), dtype=np.float32)
        padded_physio = np.zeros((target_len, 4), dtype=np.float32)
        mask = np.zeros(target_len, dtype=np.float32)

        # 填充有效数据
        padded_local[:valid_len] = visual_local[:valid_len]
        padded_physio[:valid_len] = physio_feat_4d[:valid_len]
        mask[:valid_len] = 1.0

        # [核心] 边缘填充 (Edge Padding)
        if valid_len < target_len and valid_len > 0:
            last_val = physio_feat_4d[valid_len - 1]  # 获取最后一帧 (4维)
            padded_physio[valid_len:] = last_val  # 广播填充

        subject_id_str = key.split('_', 1)[0]

        return (
            torch.FloatTensor(padded_local),
            torch.FloatTensor(visual_global),
            torch.FloatTensor(padded_physio),
            torch.FloatTensor(mask),
            torch.tensor(label, dtype=torch.long),
            subject_id_str
        )

    def __len__(self):
        return len(self.samples)


def get_loader(set_name, fold_idx=0, batch_size=config.BATCH_SIZE):
    dataset = SSTDataset(set_name, fold_idx)
    shuffle = True if set_name == 'train' else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)