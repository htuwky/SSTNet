import sys
import os
import numpy as np
import random

# 将项目根目录加入路径，确保能 import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    npy_path = config.CLIP_FEATURE_FILE

    print(f"🔍 Inspecting file: {npy_path}")

    if not os.path.exists(npy_path):
        print(f"❌ Error: File not found at {npy_path}")
        print("   Please run 'python data_process/generate_clip_features.py' first.")
        return

    try:
        # 加载 .npy (allow_pickle=True 是必须的，因为存的是字典)
        data = np.load(npy_path, allow_pickle=True).item()
        print(f"✅ Load Success! Total sequences: {len(data)}")

        if len(data) == 0:
            print("⚠️ Warning: The file is empty!")
            return

        # 随机抽取 10 个 Key
        all_keys = list(data.keys())
        sample_keys = random.sample(all_keys, min(10, len(all_keys)))

        print("\n🎲 Random Samples Check:")
        print("=" * 50)

        for key in sample_keys:
            feat = data[key]
            print(f"🔑 Key: {key}")
            print(f"📦 Shape: {feat.shape}")  # 应该是 [Seq_Len, 512]
            print(f"   Type:  {feat.dtype}")

            # 数值检查
            if feat.shape[1] != 512:
                print(f"⚠️ 维度异常! 期望 512, 实际 {feat.shape[1]}")

            mean_val = np.mean(feat)
            std_val = np.std(feat)
            print(f"   Stats: Mean={mean_val:.4f}, Std={std_val:.4f}")

            # 检查是否有 NaN 或 Inf
            if np.isnan(feat).any() or np.isinf(feat).any():
                print("❌ Error: Contains NaN or Inf!")

            # 检查是否全为 0
            if np.all(feat == 0):
                print("⚠️ Warning: Features are all ZEROS!")

            print("-" * 50)

    except Exception as e:
        print(f"❌ Error loading .npy file: {e}")


if __name__ == "__main__":
    main()