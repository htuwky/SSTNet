import os

# ================= 路径配置 =================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
IMAGE_DIR = os.path.join(DATASET_DIR, 'Images')

# [核心修改 1: TXT 文件输出路径细分]
TXT_ROOT = os.path.join(DATASET_DIR, 'TXT')
TRAIN_TXT_DIR = os.path.join(TXT_ROOT, 'Train_Valid') # 训练/验证集 TXT 路径
TEST_TXT_DIR = os.path.join(TXT_ROOT, 'Test')         # 测试集 TXT 路径

# [核心修改 2: 原始 Excel 文件 (.xlsx) 的输入路径配置]
TRAIN_FIXATIONS_DIR = os.path.join(DATASET_DIR, 'Train_Valid', 'Fixations')
# 请根据你的实际路径调整 TEST 路径
TEST_FIXATIONS_DIR = os.path.join(DATASET_DIR, 'Test', 'Fixations')

# 输出路径
OUTPUT_DIR = os.path.join(DATASET_DIR, 'output')
CLIP_TRAIN_FEATURE_FILE = os.path.join(OUTPUT_DIR, 'feature_dict_CLIP_train.npy') #
CLIP_TEST_FEATURE_FILE = os.path.join(OUTPUT_DIR, 'feature_dict_CLIP_test.npy') #

# ================= 特征提取配置 (Part 1) =================
CLIP_MODEL_NAME = "ViT-B/32"
CROP_SIZE = 224          # CLIP 输入尺寸
EXTRACT_BATCH_SIZE = 64  # 提取特征时的批次大小

# ================= 模型结构配置 (Part 2) =================
# 输入维度
INPUT_DIM = 512          # 原始 CLIP 特征维度 (Local & Global)
PHYSIO_DIM = 4           # 生理特征维度
MAX_SEQ_LEN = 32         # 序列最大长度

# [关键新增] 内部瓶颈层维度 (Bottleneck)
# 把 512 维压缩到 128 维，大幅减少参数量，防止过拟合
HIDDEN_DIM = 128         

# 时序流 (Transformer)
TEMP_LAYERS = 2          # 层数
TEMP_HEADS = 4           # [修改] 128维 / 4头 = 32每头 (适配 HIDDEN_DIM)
TEMP_FF_DIM = 512        # [修改] 128 * 4 = 512 (适配 HIDDEN_DIM)
TEMP_DROPOUT = 0.5       # [修改] 提高到 0.5 抗过拟合

# 空间流 (NetVLAD)
SPATIAL_CLUSTERS = 8     # 聚类中心数 K
SPATIAL_OUT_DIM = 128    # [修改] 输出也要对齐到 HIDDEN_DIM (原2048太大)
SPATIAL_ALPHA = 100.0    # NetVLAD 软分配系数

# 分类头 (Classifier)
# 融合后维度 = 时序流(128) + 空间流(128) = 256
CLS_HIDDEN_DIM = 64      # [修改] 分类器隐藏层维度
CLS_DROPOUT = 0.5        # [修改] 分类器 Dropout (更高一点)

# ================= 训练配置 =================
BATCH_SIZE = 64          
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3     # [修改] 加大权重衰减 (1e-4 -> 1e-2)
EPOCHS = 100
DEVICE = "cuda"          
SEED = 42                

# ================= 生理特征归一化参数 =================
SCREEN_X_MIN = 0.0
SCREEN_X_MAX = 1023.0
SCREEN_Y_MIN = 0.0
SCREEN_Y_MAX = 767.0

# 真实统计值
DUR_MIN = 0.0
DUR_MAX = 1070.8870
PUPIL_MIN = 189.0
PUPIL_MAX = 4141.0