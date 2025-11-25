import os, glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import codecs
import argparse

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, help='Path to the dataset directory')
args = parser.parse_args()

if not args.dataset_dir:
    raise ValueError("Please provide --dataset_dir argument.")

# --- 1. 路径配置 ---
input_path_val = os.path.join(args.dataset_dir, 'Train_Valid', 'Fixations', '*')
output_path = os.path.join(args.dataset_dir, 'Train_Valid', 'TXT')
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

    # 适配 Duration
    if 'FIX_DURATION' in df.columns:
        dur_col = 'FIX_DURATION'
    elif 'X_DURATI' in df.columns:
        dur_col = 'X_DURATI'
    else:
        continue

    # 适配 Pupil
    if 'FIX_PUPIL' in df.columns:
        pupil_col = 'FIX_PUPIL'
    elif 'Pupil' in df.columns:
        pupil_col = 'Pupil'
    else:
        df['FIX_PUPIL_AUTO'] = 0
        pupil_col = 'FIX_PUPIL_AUTO'

    # 提取数据
    IMAGE_list = df['IMAGE'].values.tolist()
    Fix_index_list = df['FIX_INDEX'].values
    Fix_duration_list = df[dur_col].values
    FIX_X_list = df['FIX_X'].values
    FIX_Y_list = df['FIX_Y'].values
    Fix_Pupil_list = df[pupil_col].values

    Images_in_xlsx = np.unique(IMAGE_list)

    # 获取 Images 文件夹下的所有实际图片
    Images_on_disk = []
    img_dir_root = os.path.join(args.dataset_dir, 'Images')
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
            FIX_duration = Fix_duration_list[index]
            FIX_Pupil = Fix_Pupil_list[index]

            # 越界清洗
            out_index_X = [i for i, x in enumerate(FIX_X) if x >= 1024 or x < 0]
            out_index_Y = [i for i, x in enumerate(FIX_Y) if x >= 768 or x < 0]
            out_index = list(np.unique(out_index_X + out_index_Y))

            if out_index:
                FIX_X = np.delete(FIX_X, out_index, axis=0)
                FIX_Y = np.delete(FIX_Y, out_index, axis=0)
                FIX_index = np.delete(FIX_index, out_index, axis=0)
                FIX_duration = np.delete(FIX_duration, out_index, axis=0)
                FIX_Pupil = np.delete(FIX_Pupil, out_index, axis=0)

            with codecs.open(output_file_path, 'w', 'utf-8') as output_file:
                for i in range(len(FIX_index)):
                    line = f"{FIX_index[i]},{FIX_X[i]},{FIX_Y[i]},{FIX_duration[i]},{FIX_Pupil[i]}"
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