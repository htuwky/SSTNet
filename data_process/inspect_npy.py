import sys
import os
import numpy as np
import random
import argparse  # [新增] 导入 argparse

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  #

#第三步  python data_process/inspect_npy --test
#python data_process/inspect_npy --train
#该文件检查生成的npy文件是否合格，随机抽取5个检查
#第3，是检查文件，后来训练者可以不运行
def print_stats(name, array):
    """打印数组的统计信息"""
    if array is None:
        print(f"   ❌ {name}: None")
        return

    print(f"   🔹 {name}:")
    print(f"      Shape: {array.shape}")
    print(f"      Type:  {array.dtype}")

    # 检查维度
    if array.shape[-1] != 512:
        print(f"      ⚠️ 维度异常! 期望最后一维是 512, 实际 {array.shape[-1]}")

    mean_val = np.mean(array)
    std_val = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)

    print(f"      Stats: Mean={mean_val:.4f}, Std={std_val:.4f}")
    print(f"             Range=[{min_val:.4f}, {max_val:.4f}]")

    # 检查异常值
    if np.isnan(array).any():
        print("      ❌ Error: Contains NaN!")
    if np.isinf(array).any():
        print("      ❌ Error: Contains Inf!")
    if np.all(array == 0):
        print("      ⚠️ Warning: All Zeros!")


def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="Inspect generated CLIP features NPY file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Inspect the Training/Validation feature file.')
    group.add_argument('--test', action='store_true', help='Inspect the Test feature file.')
    args = parser.parse_args()

    # --- 2. 路径确定 ---
    if args.train:
        npy_path = config.CLIP_TRAIN_FEATURE_FILE
    elif args.test:
        npy_path = config.CLIP_TEST_FEATURE_FILE
    else:
        # 理论上不会执行到这里
        return

    print(f"🔍 Inspecting file: {npy_path}")

    if not os.path.exists(npy_path):
        print(f"❌ Error: File not found at {npy_path}")
        return

    try:
        # 加载
        data = np.load(npy_path, allow_pickle=True).item()
        print(f"✅ Load Success! Total sequences: {len(data)}")

        if len(data) == 0:
            print("⚠️ Warning: The file is empty!")
            return

        # 随机抽取 5 个 Key
        all_keys = list(data.keys())
        sample_keys = random.sample(all_keys, min(5, len(all_keys)))

        print("\n🎲 Random Samples Check (SSTNet v2.0 Structure):")
        print("=" * 60)

        for key in sample_keys:
            item = data[key]
            print(f"🔑 Key: {key}")

            # 检查数据结构类型
            if isinstance(item, dict):
                # v2.0 结构: 包含 local 和 global
                if 'local' in item:
                    print_stats("Local Features (Patches)", item['local'])
                else:
                    print("   ❌ Missing 'local' key!")

                if 'global' in item:
                    print_stats("Global Feature (Context)", item['global'])
                else:
                    print("   ❌ Missing 'global' key!")

            elif isinstance(item, np.ndarray):
                # 兼容 v1.0 结构 (以防万一读了旧文件)
                print("⚠️ Warning: Detected legacy v1.0 format (Array only)")
                print_stats("Features", item)

            else:
                print(f"❌ Unknown data type: {type(item)}")

            print("-" * 60)

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        # 在调试工具中，我们可能希望看到完整的错误堆栈，但此处保持简单打印。


if __name__ == "__main__":
    main()