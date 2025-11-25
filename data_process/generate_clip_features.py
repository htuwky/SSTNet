import sys
import os

# 将项目根目录加入路径，确保能 import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # 导入配置
import glob
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


def safe_crop(image, x, y, crop_size=224):
    """
    以 (x, y) 为中心裁剪图片。自动处理越界情况 (Padding 黑色背景)。
    """
    w, h = image.size
    half = crop_size // 2

    left = x - half
    top = y - half
    right = x + half
    bottom = y + half

    # 场景A: 完全在图内
    if left >= 0 and top >= 0 and right <= w and bottom <= h:
        return image.crop((left, top, right, bottom))

    # 场景B: 越界，需要 Padding
    pad_img = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(w, right)
    src_bottom = min(h, bottom)

    if src_right > src_left and src_bottom > src_top:
        crop_part = image.crop((src_left, src_top, src_right, src_bottom))
        dst_left = max(0, -left)
        dst_top = max(0, -top)
        pad_img.paste(crop_part, (dst_left, dst_top))

    return pad_img


def build_image_map(root_dir):
    """
    递归扫描文件夹，建立 {图片名(无后缀): 完整路径} 的映射表
    """
    print(f"🔍 Scanning image directory: {root_dir} ...")
    image_map = {}
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # 获取不带后缀的文件名，作为Key
                name_no_ext = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                image_map[name_no_ext] = full_path
                count += 1
    print(f"✅ Indexed {count} images.")
    return image_map


def main():
    # 1. 准备配置
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"🔄 Loading CLIP ({config.CLIP_MODEL_NAME}) on {device}...")
    model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)
    model.eval()

    txt_dir = config.TXT_DIR
    img_dir = config.IMAGE_DIR
    output_path = config.CLIP_FEATURE_FILE

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. [核心改进] 建立图片路径映射表
    # 不管图片藏在哪个子文件夹，只要名字对得上，就能找到
    image_path_map = build_image_map(img_dir)

    # 获取所有 TXT 文件
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    print(f"📂 Found {len(txt_files)} sequence files.")

    feature_dict = {}
    error_count = 0
    skip_count = 0

    # 3. 主循环
    for txt_path in tqdm(txt_files, desc="Extracting Features"):
        filename_key = os.path.basename(txt_path).split('.txt')[0]

        # --- A. 解析文件名 ---
        # 规则：ID_ImageName (例如 002_act_001)
        try:
            subject_id, image_name_str = filename_key.split('_', 1)
        except ValueError:
            print(f"⚠️ Skipping invalid filename: {filename_key}")
            skip_count += 1
            continue

        # --- B. [改进] 查找图片文件 ---
        # 直接从地图里查，不再拼接路径猜
        if image_name_str in image_path_map:
            img_full_path = image_path_map[image_name_str]
        else:
            # print(f"⚠️ Image not found in map: {image_name_str}")
            skip_count += 1
            continue

        try:
            # --- C. 读取眼动数据 ---
            if os.path.getsize(txt_path) == 0: continue
            df = pd.read_csv(txt_path, header=None)
            coords = df.iloc[:, 1:3].values  # [[x, y], ...]

            # --- D. 裁剪与提取 ---
            img = Image.open(img_full_path).convert("RGB")

            patches = []
            for (x, y) in coords:
                patch = safe_crop(img, int(x), int(y), config.CROP_SIZE)
                patches.append(preprocess(patch))

            if not patches: continue

            # 堆叠并送入 GPU
            input_tensor = torch.stack(patches).to(device)

            # 分批提取
            features_list = []
            with torch.no_grad():
                for i in range(0, len(input_tensor), config.EXTRACT_BATCH_SIZE):
                    batch = input_tensor[i: i + config.EXTRACT_BATCH_SIZE]
                    feat = model.encode_image(batch)
                    features_list.append(feat.cpu().numpy())

            final_features = np.concatenate(features_list, axis=0).astype(np.float32)
            feature_dict[filename_key] = final_features

        except Exception as e:
            print(f"❌ Error processing {filename_key}: {e}")
            error_count += 1
            continue

    # 4. 保存
    print(f"💾 Saving features to {output_path}...")
    np.save(output_path, feature_dict)
    print(f"✅ All Done! Saved {len(feature_dict)} sequences. (Skipped: {skip_count}, Errors: {error_count})")


if __name__ == "__main__":
    main()