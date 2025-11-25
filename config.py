# config.py
import os

# ================= 路径配置 =================
# 项目根目录 (假设 config.py 在根目录下)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'Images')
TXT_DIR = os.path.join(DATASET_DIR, 'Train_Valid', 'TXT')

# 输出路径
OUTPUT_DIR = os.path.join(DATASET_DIR, 'output')
CLIP_FEATURE_FILE = os.path.join(OUTPUT_DIR, 'feature_dict_CLIP.npy')

# ================= 特征提取配置 (Part 1) =================
CLIP_MODEL_NAME = "ViT-B/32"
CROP_SIZE = 224          # CLIP 输入尺寸
EXTRACT_BATCH_SIZE = 64  # 提取特征时的批次大小

# ================= 模型结构配置 (Part 2) =================
# 输入
INPUT_DIM = 512          # CLIP 特征维度
PHYSIO_DIM = 4           # 生理特征维度 (x, y, duration, pupil)
MAX_SEQ_LEN = 32         # [关键] 序列最大长度 (不足补0，超过截断)

# 时序流 (Transformer)
TEMP_LAYERS = 2          # 层数
TEMP_HEADS = 8           # 注意力头数
TEMP_FF_DIM = 2048       # 前馈层维度
TEMP_DROPOUT = 0.2       # 防止过拟合

# 空间流 (NetVLAD)
SPATIAL_CLUSTERS = 8     # 聚类中心数 K
SPATIAL_OUT_DIM = 512    # 投影回的维度
SPATIAL_ALPHA = 100.0    # [新增] NetVLAD 软分配系数

# 分类头 (Classifier)
CLS_HIDDEN_DIM = 256     # [新增] 分类器隐藏层维度
CLS_DROPOUT = 0.5        # [新增] 分类器 Dropout (通常比 Transformer 高)

# ================= 训练配置 =================
BATCH_SIZE = 64          # 训练时的 Batch Size
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
DEVICE = "cuda"          # 自动检测逻辑在 train.py 里写，这里只是默认值
SEED = 42                # 随机种子

# 屏幕尺寸
SCREEN_X_MIN = 0.0
SCREEN_X_MAX = 1023.0
SCREEN_Y_MIN = 0.0
SCREEN_Y_MAX = 767.0

# 注视点 & 瞳孔大小的合理范围 (用于数据清洗)
# 真实统计值 (已填入你刚才跑出的结果)
DUR_MIN = 0.0
DUR_MAX = 1071.0   # 超过1秒的注视被视为异常长，截断
PUPIL_MIN = 189.0
PUPIL_MAX = 4141.0 # 瞳孔最大值