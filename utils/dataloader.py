import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SSTDataset(Dataset):
    def __init__(self, set_name, fold_idx=0):
        self.set_name = set_name
        self.max_len = config.MAX_SEQ_LEN

        # [修改后] config.CLIP_TRAIN_FEATURE_FILE
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
        visual_local = data_pack['local']  # [Seq, 512] (numpy array)
        visual_global = data_pack['global']  # [1, 512] (numpy array)

        # --- B. 获取生理特征 ---
        try:
            df = pd.read_csv(txt_path, header=None)
            raw_data = df.iloc[:, 1:5].values.astype(np.float32)

            min_vec = np.array([config.SCREEN_X_MIN, config.SCREEN_Y_MIN, config.DUR_MIN, config.PUPIL_MIN],
                               dtype=np.float32)
            max_vec = np.array([config.SCREEN_X_MAX, config.SCREEN_Y_MAX, config.DUR_MAX, config.PUPIL_MAX],
                               dtype=np.float32)

            raw_data = np.clip(raw_data, min_vec, max_vec)
            norm_0_1 = (raw_data - min_vec) / (max_vec - min_vec + 1e-6)
            physio_feat = norm_0_1 * 2 - 1

        except Exception:
            # 如果读取失败，给一个全 0 的特征，长度与视觉特征一致
            physio_feat = np.zeros((visual_local.shape[0], config.PHYSIO_DIM), dtype=np.float32)

        # ==============================================================================
        # 【核心修改 - 已修复位置】特征级数据增强 (仅在训练模式下)
        # 必须在 Padding 之前对原始特征进行操作！
        # ==============================================================================
        if self.set_name == 'train':
            # 1. 特征加噪 (Feature Noise)
            noise_level = 0.015  # 噪声强度
            noise = np.random.normal(0, noise_level, visual_local.shape).astype(np.float32)
            visual_local = visual_local + noise

            # 2. 时序随机丢弃 (Temporal Masking)
            drop_prob = 0.15  # 15% 的帧被随机丢弃
            seq_len_origin = visual_local.shape[0]  # 使用原始序列长度

            # 生成掩码：保留(True/1), 丢弃(False/0)
            drop_mask = (np.random.rand(seq_len_origin) > drop_prob).astype(np.float32)
            # 扩展维度以匹配特征 [Seq, 1]
            drop_mask = drop_mask[:, np.newaxis]

            # 直接应用到原始视觉特征上
            visual_local = visual_local * drop_mask
            # 【重要】生理特征也同步遮蔽，保证信息一致性
            physio_feat = physio_feat * drop_mask
        # ==============================================================================

        # --- C. 统一长度 (Padding) ---
        # 此时的 visual_local 和 physio_feat 已经是增强过的数据了
        seq_len = visual_local.shape[0]
        target_len = self.max_len

        padded_local = np.zeros((target_len, config.INPUT_DIM), dtype=np.float32)
        padded_physio = np.zeros((target_len, config.PHYSIO_DIM), dtype=np.float32)
        mask = np.zeros(target_len, dtype=np.float32)

        # 计算有效长度（处理长序列截断）
        valid_len = min(seq_len, target_len)

        # 填充数据（如果 seq_len > target_len，这里会自动截断）
        padded_local[:valid_len] = visual_local[:valid_len]
        padded_physio[:valid_len] = physio_feat[:valid_len]
        mask[:valid_len] = 1.0

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