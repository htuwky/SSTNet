import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt  # 用于画图分析

#该文件用于判断  生成多少个注视点,根据判断，32个为最佳，大部分不够。填0补充
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()

    # 1. 确定 TXT 路径
    txt_dir = os.path.join(args.dataset_dir, 'Train_Valid', 'TXT')
    if not os.path.exists(txt_dir):
        print(f"❌ Error: Directory not found: {txt_dir}")
        print("Please run split_fix.py first!")
        return

    # 2. 获取所有 TXT 文件
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    print(f"📂 Found {len(txt_files)} sequence files. Analyzing...")

    if len(txt_files) == 0:
        print("⚠️ No .txt files found.")
        return

    # 3. 统计行数
    lengths = []
    # 记录超长文件的名字，方便后续检查
    long_sequences = []

    for f_path in tqdm(txt_files):
        with open(f_path, 'r', encoding='utf-8') as f:
            # 统计非空行数
            lines = [line.strip() for line in f if line.strip()]
            count = len(lines)
            lengths.append(count)

            if count > 50:  # 记录一下特别长的
                long_sequences.append((os.path.basename(f_path), count))

    # 4. 计算统计指标
    lengths = np.array(lengths)

    print("\n" + "=" * 40)
    print("📊 Fixation Sequence Statistics")
    print("=" * 40)
    print(f"Total Sequences: {len(lengths)}")
    print(f"Min Length:      {np.min(lengths)}")
    print(f"Max Length:      {np.max(lengths)}")
    print(f"Mean Length:     {np.mean(lengths):.2f}")
    print(f"Median Length:   {np.median(lengths)}")
    print("-" * 40)

    # 5. 覆盖率分析 (帮你做决策)
    print("💡 Coverage Analysis (How many sequences fit in X?):")
    for threshold in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
        coverage = np.sum(lengths <= threshold) / len(lengths) * 100
        print(f"  Seq_Len <= {threshold}: {coverage:.2f}%")

    print("-" * 40)

    # 6. 打印分位数 (更科学的参考)
    print(f"90% of data is <= {np.percentile(lengths, 90):.0f}")
    print(f"95% of data is <= {np.percentile(lengths, 95):.0f}")
    print(f"99% of data is <= {np.percentile(lengths, 99):.0f}")

    if long_sequences:
        print("\n⚠️ Extreme Outliers (Top 3):")
        # 按长度降序排
        long_sequences.sort(key=lambda x: x[1], reverse=True)
        for name, count in long_sequences[:3]:
            print(f"  {name}: {count} fixations")

    # 7. (可选) 画个分布图
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=range(0, max(lengths) + 2), alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Fixation Sequence Lengths')
        plt.xlabel('Number of Fixations')
        plt.ylabel('Count')
        plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=1,
                    label=f'Mean: {np.mean(lengths):.1f}')
        plt.axvline(np.percentile(lengths, 95), color='green', linestyle='dashed', linewidth=1, label='95% Percentile')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)

        save_path = os.path.join(args.dataset_dir, 'fixation_length_dist.png')
        plt.savefig(save_path)
        print(f"\n📈 Histogram saved to: {save_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save plot (matplotlib error): {e}")


if __name__ == "__main__":
    main()