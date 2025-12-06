import os, glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import codecs
import argparse
import sys

#1.第一步需要python split_fix.py --test
# python split_fix.py --train

# 将项目根目录加入路径，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config #
# --- 参数解析 ---
# --- 参数解析 ---
parser = argparse.ArgumentParser()
# [修改] 引入互斥组来选择模式
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true', help='Process files from the Training Fixations directory, output to Train_Valid/TXT.')
group.add_argument('--test', action='store_true', help='Process files from the Testing Fixations directory, output to Test/TXT.')
args = parser.parse_args()

# --- 1. 路径配置 ---
if args.train:
    # [选择训练集路径]
    input_fixation_dir = config.TRAIN_FIXATIONS_DIR
    output_path = config.TRAIN_TXT_DIR
    print(f"🚀 Processing Training Fixations from: {input_fixation_dir}")
elif args.test:
    # [选择测试集路径]
    input_fixation_dir = config.TEST_FIXATIONS_DIR
    output_path = config.TEST_TXT_DIR
    print(f"🚀 Processing Testing Fixations from: {input_fixation_dir}")
else:
    raise ValueError("Internal Error: Must specify either --train or --test mode.")


input_path_val = os.path.join(input_fixation_dir, '*')
os.makedirs(output_path, exist_ok=True)

# 获取所有 Excel 文件
excel_files = glob.glob(input_path_val)
print(f"📂 Found {len(excel_files)} Excel files in Train_Valid/Fixations. Start processing...")

# --- 计数器 ---
success_count = 0
fail_count = 0

# --- 2. 主处理循环 ---
for f_path in tqdm(excel_files, desc="Processing Files"):
    subject_index = os.path.basename(f_path).split('.')[0]

    try:
        # 尝试读取 Excel
        df = pd.read_excel(io=f_path, engine='openpyxl')  # 显式指定引擎，如果没装会直接在这里报错提示更清楚
        success_count += 1
    except Exception as e:
        # 如果读取失败，打印红色错误信息，并增加失败计数
        print(f"\n❌ Error reading {os.path.basename(f_path)}: {e}")
        fail_count += 1
        continue

    # --- 3. 读取关键列 ---
    required_cols = ['IMAGE', 'FIX_INDEX', 'FIX_X', 'FIX_Y']
    # 稍微放宽检查，防止部分文件表头大小写不一致
    df.columns = [c.strip() for c in df.columns]  # 去除列名空格

    if not all(col in df.columns for col in required_cols):
        print(f"\n⚠️ Warning: Missing standard columns in {os.path.basename(f_path)}. Skipping.")
        continue

    # # 适配 Duration
    # if 'FIX_DURATION' in df.columns:
    #     dur_col = 'FIX_DURATION'
    # elif 'X_DURATI' in df.columns:
    #     dur_col = 'X_DURATI'
    # else:
    #     continue
    #
    # # 适配 Pupil
    # if 'FIX_PUPIL' in df.columns:
    #     pupil_col = 'FIX_PUPIL'
    # elif 'Pupil' in df.columns:
    #     pupil_col = 'Pupil'
    # else:
    #     df['FIX_PUPIL_AUTO'] = 0
    #     pupil_col = 'FIX_PUPIL_AUTO'

    # 提取数据
    IMAGE_list = df['IMAGE'].values.tolist()
    Fix_index_list = df['FIX_INDEX'].values
    # Fix_duration_list = df[dur_col].values
    FIX_X_list = df['FIX_X'].values
    FIX_Y_list = df['FIX_Y'].values
    # Fix_Pupil_list = df[pupil_col].values

    Images_in_xlsx = np.unique(IMAGE_list)

    # 获取 Images 文件夹下的所有实际图片
    Images_on_disk = []
    img_dir_root = config.IMAGE_DIR  # # 使用 config.py 中集中配置的路径
    if os.path.exists(img_dir_root):
        for home, dirs, files in os.walk(img_dir_root):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    Images_on_disk.append(filename)

    # --- 4. 按图片拆分并保存 TXT ---
    for image_file in Images_on_disk:
        image_name_no_ext = image_file.split('.')[0]
        folder_name = f"{subject_index}_{image_name_no_ext}"
        output_file_path = os.path.join(output_path, f'{folder_name}.txt')

        if image_file not in Images_in_xlsx:
            with codecs.open(output_file_path, 'w', 'utf-8') as output_file:
                pass
        else:
            index = [i for i, x in enumerate(IMAGE_list) if x == image_file]

            FIX_index = Fix_index_list[index]
            FIX_X = np.floor(FIX_X_list[index]).astype(np.int64)
            FIX_Y = np.floor(FIX_Y_list[index]).astype(np.int64)
            # FIX_duration = Fix_duration_list[index]
            # FIX_Pupil = Fix_Pupil_list[index]

            # 越界清洗
            out_index_X = [i for i, x in enumerate(FIX_X) if x > config.SCREEN_X_MAX or x < config.SCREEN_X_MIN]  #
            out_index_Y = [i for i, x in enumerate(FIX_Y) if x > config.SCREEN_Y_MAX or x < config.SCREEN_Y_MIN]  #
            out_index = list(np.unique(out_index_X + out_index_Y))

            if out_index:
                FIX_X = np.delete(FIX_X, out_index, axis=0)
                FIX_Y = np.delete(FIX_Y, out_index, axis=0)
                FIX_index = np.delete(FIX_index, out_index, axis=0)
                # FIX_duration = np.delete(FIX_duration, out_index, axis=0)
                # FIX_Pupil = np.delete(FIX_Pupil, out_index, axis=0)

            with codecs.open(output_file_path, 'w', 'utf-8') as output_file:
                for i in range(len(FIX_index)):
                    line = f"{FIX_index[i]},{FIX_X[i]},{FIX_Y[i]}"
                    output_file.write(line)
                    output_file.write('\n')

# --- 5. 打印最终统计结果 ---
print("\n" + "=" * 50)
print(f"📊 Processing Summary:")
print(f"✅ Successful: {success_count} files")
print(f"❌ Failed:     {fail_count} files")
print("=" * 50)

if success_count > 0:
    print(f"TXT files generated in {output_path}")
else:
    print("⚠️ No files were processed successfully. Please check errors above.")