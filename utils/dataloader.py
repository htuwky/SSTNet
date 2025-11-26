import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys

# 将项目根目录加入路径，确保能 import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SSTDataset(Dataset):
    def __init__(self, set_name, fold_idx=0):
        """
        Args:
            set_name: 'train' 或 'val'
            fold_idx: 当前是第几折交叉验证 (0-3)
        """
        self.set_name = set_name
        self.max_len = config.MAX_SEQ_LEN

        # 1. 加载视觉特征大字典 (.npy)
        # 格式: {'Subject_Image': [Seq_Len, 512]}
        print(f"🔄 [{set_name.upper()}] Loading visual features from {config.CLIP_FEATURE_FILE} ...")
        try:
            self.visual_data = np.load(config.CLIP_FEATURE_FILE, allow_pickle=True).item()
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Feature file not found! Please run generate_clip_features.py first.")

        # 2. 获取数据列表 (根据 Train_Valid.xlsx 和 fold_idx 划分)
        self.samples = self._split_dataset(fold_idx)
        print(f"✅ {set_name.upper()} set loaded: {len(self.samples)} samples.")

    def _split_dataset(self, fold_idx):
        """
        读取 Train_Valid.xlsx，根据 fold_idx 将受试者划分为训练集或验证集。
        并找到这些受试者对应的所有 TXT 数据文件。
        """
        excel_path = os.path.join(config.DATASET_DIR, 'Train_Valid.xlsx')
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"❌ Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path)
        folds = ['Set_0', 'Set_1', 'Set_2', 'Set_3']

        # 确定当前折的目标受试者 ID 列表
        val_col = folds[fold_idx]
        train_cols = [f for f in folds if f != val_col]

        target_ids = []
        if self.set_name == 'val':
            # 验证集: 只取 val_col 列的 ID
            raw_ids = df[val_col].dropna().values
            target_ids = [str(int(i)).zfill(3) for i in raw_ids]
        else:
            # 训练集: 合并其他 3 列的 ID
            for col in train_cols:
                raw_ids = df[col].dropna().values
                target_ids.extend([str(int(i)).zfill(3) for i in raw_ids])

        # 遍历 TXT 文件夹，匹配属于 target_ids 的文件
        sample_list = []
        txt_folder = config.TXT_DIR
        if not os.path.exists(txt_folder):
            raise FileNotFoundError(f"❌ TXT folder not found: {txt_folder}")

        all_files = os.listdir(txt_folder)

        for f in all_files:
            if not f.endswith('.txt'): continue

            # 文件名解析: SubjectID_ImageName.txt
            filename_no_ext = f.split('.txt')[0]
            try:
                # 假设规则: ID_ImageName (只切第一个下划线)
                subj_id, _ = filename_no_ext.split('_', 1)
            except ValueError:
                continue

            # 匹配 ID
            if subj_id in target_ids:
                # 标签规则: ID < 200 为 HC(0), >= 200 为 SZ(1)
                label = 0 if int(subj_id) < 200 else 1

                # [关键] 必须同时在 .npy 里有视觉特征才算有效数据
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
        subject_id_str = key.split('_', 1)[0]
        # --- A. 获取视觉特征 (Visual) ---
        # Shape: [Seq_Len, 512]
        visual_feat = self.visual_data[key]

        # --- B. 获取生理特征 (Physio) 并映射 ---
        try:
            # 读取 TXT: Index, X, Y, Duration, Pupil
            df = pd.read_csv(txt_path, header=None)

            # 取第 1,2,3,4 列 (X, Y, Duration, Pupil)
            # Shape: [Seq_Len, 4]
            raw_data = df.iloc[:, 1:5].values.astype(np.float32)

            # 1. 准备边界值 (从 config 读取，顺序必须对应)
            min_vec = np.array([config.SCREEN_X_MIN, config.SCREEN_Y_MIN, config.DUR_MIN, config.PUPIL_MIN],
                               dtype=np.float32)
            max_vec = np.array([config.SCREEN_X_MAX, config.SCREEN_Y_MAX, config.DUR_MAX, config.PUPIL_MAX],
                               dtype=np.float32)

            # 2. 截断离群值 (Clip) - 保护归一化不被极值破坏
            raw_data = np.clip(raw_data, min_vec, max_vec)

            # 3. Min-Max 归一化到 0~1
            # 加 1e-6 防止除以 0
            norm_0_1 = (raw_data - min_vec) / (max_vec - min_vec + 1e-6)

            # 4. 映射到 -1~1 (与 CLIP 特征对齐)
            physio_feat = norm_0_1 * 2 - 1

        except Exception:
            # 容错处理：如果读取失败，给全0 (理论上前面过滤过空文件，不会触发)
            physio_feat = np.zeros((visual_feat.shape[0], config.PHYSIO_DIM), dtype=np.float32)

        # --- C. 统一长度 & 生成 Mask ---
        seq_len = visual_feat.shape[0]
        target_len = self.max_len

        # 初始化容器 (全 0 代表 Padding)
        padded_visual = np.zeros((target_len, config.INPUT_DIM), dtype=np.float32)
        padded_physio = np.zeros((target_len, config.PHYSIO_DIM), dtype=np.float32)
        mask = np.zeros(target_len, dtype=np.float32)  # 0=Pad, 1=Real

        # 截取有效长度 (防止数据超过 32)
        valid_len = min(seq_len, target_len)

        # 填入真实数据
        padded_visual[:valid_len] = visual_feat[:valid_len]
        # 如果 txt 行数少于 npy (罕见)，这里会自动切片匹配；如果多于，也会截断
        # 为了安全，取两者最小行数作为填充长度
        fill_len = min(valid_len, physio_feat.shape[0])
        padded_physio[:fill_len] = physio_feat[:fill_len]

        # 标记 Mask (有数据的地方设为 1)
        mask[:fill_len] = 1.0

        # 返回 Tensor
        return (
            torch.FloatTensor(padded_visual),  # [32, 512]
            torch.FloatTensor(padded_physio),  # [32, 4]
            torch.FloatTensor(mask),  # [32]
            torch.tensor(label, dtype=torch.long),  # Scalar (0/1)
            subject_id_str  # 5. [新增] 受试者ID (字符串)
        )

    def __len__(self):
        return len(self.samples)


def get_loader(set_name, fold_idx=0, batch_size=config.BATCH_SIZE):
    """
    获取 DataLoader 的便捷函数
    """
    dataset = SSTDataset(set_name, fold_idx)

    # 训练集打乱，验证集不打乱
    shuffle = True if set_name == 'train' else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows下建议设为0避免多进程报错，Linux可设为4
        pin_memory=True
    )
    return loader


# --- 简单的测试块 (直接运行此脚本可测试) ---
if __name__ == "__main__":
    print("🚀 Testing DataLoader...")
    try:
        # 尝试加载第 0 折的训练集，取 2 个样本
        loader = get_loader('train', fold_idx=0, batch_size=2)

        for vis, phy, mask, label in loader:
            print(f"  Visual Shape: {vis.shape} (Expect [2, {config.MAX_SEQ_LEN}, 512])")
            print(f"  Physio Shape: {phy.shape} (Expect [2, {config.MAX_SEQ_LEN}, 4])")
            print(f"  Mask Shape:   {mask.shape} (Expect [2, {config.MAX_SEQ_LEN}])")
            print(f"  Label:        {label} (Expect [0/1, 0/1])")

            # 检查归一化范围
            print(f"  Physio Min/Max: {phy.min():.2f} / {phy.max():.2f} (Expect approx -1 to 1)")

            # 检查 Mask 是否生效 (打印第一个样本的有效长度)
            valid_len = mask[0].sum().item()
            print(f"  Sample 0 Valid Length: {int(valid_len)}")
            break

        print("✅ DataLoader test passed!")
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback

        traceback.print_exc()