import sys
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# [修改] 移除 try/except 块，直接导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

#第五步  python check_data_stats.py
#第五步是为了确定标准/归一化数据进行的数据分析，后来训练者可根据自己的数据集进行修改
#若只是用本数据集进行修训练，第3，4，5步均可不用运行
# [修改] 使用新的 config 变量，只分析训练集数据
TXT_DIR = config.TRAIN_TXT_DIR
NPY_PATH = config.CLIP_TRAIN_FEATURE_FILE


def analyze_physio_data(txt_dir):
    print(f"\n🔍 [1/2] Scanning Physiological Data (.txt) in: {txt_dir}")

    files = glob.glob(os.path.join(txt_dir, '*.txt'))
    if len(files) == 0:
        print("❌ No TXT files found!")
        return None

    # 使用列表暂存 (比 np.append 快得多)
    all_x, all_y, all_dur, all_pupil = [], [], [], []
    valid_files = 0

    for f in tqdm(files, desc="Reading ALL TXT Files"):
        if os.path.getsize(f) == 0: continue
        try:
            # 格式: Index, X, Y, Duration, Pupil
            df = pd.read_csv(f, header=None)
            if df.shape[1] < 5: continue

            # 显式转换为 float32 节省内存
            all_x.extend(df.iloc[:, 1].values.astype(np.float32))
            all_y.extend(df.iloc[:, 2].values.astype(np.float32))
            all_dur.extend(df.iloc[:, 3].values.astype(np.float32))
            all_pupil.extend(df.iloc[:, 4].values.astype(np.float32))
            valid_files += 1
        except Exception:
            continue

    print(f"✅ Processed {valid_files} valid files.")

    # 转换为 Numpy 数组
    stats = {
        "X-Coordinate": np.array(all_x),
        "Y-Coordinate": np.array(all_y),
        "Duration (ms)": np.array(all_dur),
        "Pupil Size": np.array(all_pupil)
    }

    results = {}
    print("-" * 75)
    print(f"{'Feature':<15} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 75)

    for name, data in stats.items():
        if len(data) == 0: continue
        _min, _max = np.min(data), np.max(data)
        _mean, _std = np.mean(data), np.std(data)
        print(f"{name:<15} | {_min:<10.4f} | {_max:<10.4f} | {_mean:<10.4f} | {_std:<10.4f}")
        results[name] = {'min': _min, 'max': _max, 'mean': _mean, 'std': _std}
    print("-" * 75)
    return results


def analyze_clip_features(npy_path):
    print(f"\n🔍 [2/2] Analyzing ALL CLIP Features (.npy) from: {npy_path}")

    if not os.path.exists(npy_path):
        print("❌ Feature file not found!")
        return None

    try:
        # 加载字典
        data = np.load(npy_path, allow_pickle=True).item()
        total_seqs = len(data)
        print(f"✅ Dictionary loaded. Containing {total_seqs} sequences.")

        print("⏳ Stacking all features for rigorous analysis...")
        all_feats = []
        for k in tqdm(data.keys(), desc="Stacking"):

            item = data[k]
            # 兼容 v2.0 结构 (如果它是字典，尝试堆叠 local 和 global 特征)
            if isinstance(item, dict):
                if 'local' in item:
                    all_feats.append(item['local'])
                # global 是 [1, 512]，也堆叠进去
                if 'global' in item:
                    all_feats.append(item['global'])
            elif isinstance(item, np.ndarray):
                all_feats.append(item)

        if not all_feats:
            print("⚠️ No valid features found in the NPY file.")
            return None

        # 堆叠成超级大矩阵: [Total_Points, 512]
        large_mat = np.vstack(all_feats)

        print(f"📊 Global Matrix Shape: {large_mat.shape}")

        _min = np.min(large_mat)
        _max = np.max(large_mat)
        _mean = np.mean(large_mat)
        _std = np.std(large_mat)

        print("-" * 75)
        print(f"CLIP Feature Statistics (All {large_mat.shape[0]} points)")
        print(f"Min:  {_min:.6f}")
        print(f"Max:  {_max:.6f}")
        print(f"Mean: {_mean:.6f} (Should be close to 0)")
        print(f"Std:  {_std:.6f}")
        print("-" * 75)

        return {'min': _min, 'max': _max, 'mean': _mean, 'std': _std}

    except Exception as e:
        print(f"❌ Error reading/processing .npy: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 1. 统计生理数据
    physio_stats = analyze_physio_data(TXT_DIR)

    # 2. 统计 CLIP 数据
    clip_stats = analyze_clip_features(NPY_PATH)

    # 3. 给出严谨建议
    if physio_stats and clip_stats:
        print("\n💡 [Rigorous Normalization Strategy]")
        print("=" * 60)

        # 检查 CLIP 分布
        print(f"1. CLIP Visual Embeddings:")
        print(f"   Range: [{clip_stats['min']:.4f}, {clip_stats['max']:.4f}]")
        print(f"   Dist : Mean~{clip_stats['mean']:.2f}, Std~{clip_stats['std']:.2f}")

        if abs(clip_stats['mean']) < 0.1 and 0.8 < clip_stats['std'] < 1.2:
            target_norm_desc = "Standardization (Mean=0, Std=1)"
            method = "z-score"
        else:
            target_norm_desc = "Min-Max Scaling (-1 to 1)"
            method = "minmax"

        print(f"   -> Target Distribution: {target_norm_desc}")

        print(f"\n2. Physiological Data Normalization Parameters (Copy to config.py):")
        print(f"   (Using 3-Sigma rule to handle outliers for robust scaling)")

        for name, stat in physio_stats.items():
            # 使用 3-Sigma 确定鲁棒的边界，防止极值破坏归一化
            robust_max = stat['mean'] + 3 * stat['std']
            robust_min = max(0, stat['mean'] - 3 * stat['std'])  # 物理量通常非负

            # 如果极值没有偏离太远，就用真实极值
            final_max = min(stat['max'], robust_max) if stat['max'] > robust_max * 1.5 else stat['max']
            final_min = stat['min']  # 最小值通常比较稳定

            print(f"   🔹 {name}:")
            print(f"      CONFIG_MIN = {final_min:.4f}")
            print(f"      CONFIG_MAX = {final_max:.4f}")
            if stat['max'] > robust_max:
                print(
                    f"      (Note: Original Max was {stat['max']:.4f}, clipped to {final_max:.4f} to exclude outliers)")