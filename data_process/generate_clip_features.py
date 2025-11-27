import sys
import os
import argparse  # [新增] 导入 argparse

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  #
import glob
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

#2第二步，python data_process/generate_clip_features.py --test
# python data_process/generate_clip_features.py --train
def safe_crop(image, x, y, crop_size=224):
    """根据注视点坐标安全裁剪图像块（包含边界填充逻辑）"""
    w, h = image.size
    half = crop_size // 2
    left, top, right, bottom = x - half, y - half, x + half, y + half

    if left >= 0 and top >= 0 and right <= w and bottom <= h:
        return image.crop((left, top, right, bottom))

    pad_img = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(w, right), min(h, bottom)

    if src_right > src_left and src_bottom > src_top:
        crop_part = image.crop((src_left, src_top, src_right, src_bottom))
        pad_img.paste(crop_part, (max(0, -left), max(0, -top)))
    return pad_img


def build_image_map(root_dir):
    """递归扫描文件夹，建立 {图片名: 完整路径} 的映射表"""
    print(f"🔍 Scanning image directory: {root_dir} ...")
    image_map = {}
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # 使用 os.path.splitext 确保只去除最后一个扩展名
                name_no_ext = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                image_map[name_no_ext] = full_path
                count += 1
    print(f"✅ Indexed {count} images.")
    return image_map


def main():
    # --- 1. 参数解析与路径确定 ---
    parser = argparse.ArgumentParser(description="Generate CLIP features for train or test set.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Extract features for the Training/Validation set.')
    group.add_argument('--test', action='store_true', help='Extract features for the Test set.')
    args = parser.parse_args()

    if args.train:
        txt_dir = config.TRAIN_TXT_DIR  #
        output_path = config.CLIP_TRAIN_FEATURE_FILE  #
        mode_name = "Training/Validation"
    elif args.test:
        txt_dir = config.TEST_TXT_DIR  #
        output_path = config.CLIP_TEST_FEATURE_FILE  #
        mode_name = "Testing"
    else:
        # should not happen
        return

    device = config.DEVICE if torch.cuda.is_available() else "cpu"  #
    img_dir = config.IMAGE_DIR  #

    print(f"🚀 Starting CLIP Feature Extraction for {mode_name} set.")
    print(f"   Reading TXT files from: {txt_dir}")
    print(f"   Output NPY to: {output_path}")

    # --- 2. 初始化模型 ---
    print(f"🔄 Loading CLIP ({config.CLIP_MODEL_NAME}) on {device}...")  #
    model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)  #
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. 建立图片地图
    image_path_map = build_image_map(img_dir)

    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    print(f"📂 Found {len(txt_files)} sequence files.")

    feature_dict = {}
    error_count = 0
    skip_count = 0

    # --- 4. 提取循环（双流逻辑完整保留） ---
    for txt_path in tqdm(txt_files, desc="Extracting Local+Global"):
        filename_key = os.path.basename(txt_path).split('.txt')[0]

        try:
            # 解析文件名
            try:
                # [核心修改：使用条件逻辑区分 Train/Test 解析]
                if args.train:
                    # 训练/验证集: Subject ID 通常没有下划线 (e.g., T001_A01)
                    subject_id, image_name_str = filename_key.split('_', 1)
                elif args.test:
                    # 测试集: Test_XXX_ImageName (ImageName 可能含下划线，例如 cat_012)

                    # 1. 将文件名通过所有下划线完全分割
                    parts = filename_key.split('_')

                    # 2. 检查长度是否满足 Test_XXX_... 的格式
                    if len(parts) >= 3:
                        # Subject ID 总是前两部分 (Test_000)
                        subject_id = f"{parts[0]}_{parts[1]}"

                        # 图片名是第三部分到末尾的所有内容，重新以下划线连接起来
                        image_name_str = '_'.join(parts[2:])
                    else:
                        # 如果不满足 Test_XXX_... 的格式，跳过
                        skip_count += 1
                        continue

            except ValueError:
                # 如果文件名中连一个下划线都没有，则跳过
                skip_count += 1;
                continue

            # 2. 从地图里查找图片路径
            if image_name_str in image_path_map:
                img_full_path = image_path_map[image_name_str]
            else:
                # 找不到图片，跳过
                skip_count += 1;
                continue

            # 读取数据
            if os.path.getsize(txt_path) == 0: continue
            df = pd.read_csv(txt_path, header=None)
            coords = df.iloc[:, 1:3].values

            img = Image.open(img_full_path).convert("RGB")

            # === 提取全局特征 (Global) ===
            global_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                global_feat = model.encode_image(global_tensor).cpu().numpy()
                # CLIP 默认归一化
                global_feat = global_feat / np.linalg.norm(global_feat, axis=1, keepdims=True)

            # === 提取局部特征 (Local) ===
            patches = []
            for (x, y) in coords:
                patch = safe_crop(img, int(x), int(y), config.CROP_SIZE)  #
                patches.append(preprocess(patch))

            if not patches: continue

            input_tensor = torch.stack(patches).to(device)
            local_feats_list = []
            with torch.no_grad():
                # 批量提取局部特征
                for i in range(0, len(input_tensor), config.EXTRACT_BATCH_SIZE):  #
                    batch = input_tensor[i: i + config.EXTRACT_BATCH_SIZE]
                    feat = model.encode_image(batch)
                    local_feats_list.append(feat.cpu().numpy())

            local_feats = np.concatenate(local_feats_list, axis=0).astype(np.float32)
            # CLIP 默认归一化
            local_feats = local_feats / np.linalg.norm(local_feats, axis=1, keepdims=True)

            # === 存入字典 ===
            feature_dict[filename_key] = {
                'local': local_feats,
                'global': global_feat
            }

        except Exception as e:
            # print(f"Error: {e}")
            error_count += 1;
            continue

    print(f"💾 Saving features to {output_path}...")
    np.save(output_path, feature_dict)
    print(f"✅ Done! Saved {len(feature_dict)} sequences. (Skipped: {skip_count}, Errors: {error_count})")


if __name__ == "__main__":
    main()