import sys
import os

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import glob
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


def safe_crop(image, x, y, crop_size=224):
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
    """
    [核心修复] 递归扫描文件夹，建立 {图片名: 完整路径} 的映射表
    解决图片在子文件夹里找不到的问题。
    """
    print(f"🔍 Scanning image directory: {root_dir} ...")
    image_map = {}
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                name_no_ext = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                image_map[name_no_ext] = full_path
                count += 1
    print(f"✅ Indexed {count} images.")
    return image_map


def main():
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"🔄 [v2.1 Fixed] Loading CLIP ({config.CLIP_MODEL_NAME}) on {device}...")
    model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)
    model.eval()

    txt_dir = config.TXT_DIR
    img_dir = config.IMAGE_DIR
    output_path = config.CLIP_FEATURE_FILE

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. [关键修复] 先建立图片地图
    image_path_map = build_image_map(img_dir)

    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    print(f"📂 Found {len(txt_files)} sequence files.")

    feature_dict = {}
    error_count = 0
    skip_count = 0

    for txt_path in tqdm(txt_files, desc="Extracting Local+Global"):
        filename_key = os.path.basename(txt_path).split('.txt')[0]

        try:
            # 解析文件名
            try:
                subject_id, image_name_str = filename_key.split('_', 1)
            except ValueError:
                skip_count += 1;
                continue

            # 2. [关键修复] 从地图里查找图片路径，而不是直接拼接
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

            # === 3. 提取全局特征 (Global) ===
            global_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                global_feat = model.encode_image(global_tensor).cpu().numpy()

            # === 4. 提取局部特征 (Local) ===
            patches = []
            for (x, y) in coords:
                patch = safe_crop(img, int(x), int(y), config.CROP_SIZE)
                patches.append(preprocess(patch))

            if not patches: continue

            input_tensor = torch.stack(patches).to(device)
            local_feats_list = []
            with torch.no_grad():
                for i in range(0, len(input_tensor), config.EXTRACT_BATCH_SIZE):
                    batch = input_tensor[i: i + config.EXTRACT_BATCH_SIZE]
                    feat = model.encode_image(batch)
                    local_feats_list.append(feat.cpu().numpy())

            local_feats = np.concatenate(local_feats_list, axis=0).astype(np.float32)

            # === 5. 存入字典 ===
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