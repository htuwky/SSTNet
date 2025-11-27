# SSTNet: Spatial-Spectral Temporal Network for Eye-Tracking Classification



**SSTNet** 是一个专为眼动追踪数据（Eye-Tracking Data）设计的多模态深度学习分类框架。

本项目旨在通过深度融合受试者的**注视行为（时序与生理特征）和注视内容（空间与视觉特征）**，辅助诊断精神疾病（例如，区分精神分裂症患者 SZ 与健康对照组 HC）。

其核心设计哲学是 **"Look where humans look"**：模型不再盲目处理整张图像，而是精确地聚焦于人眼关注的局部区域，并结合全局上下文进行推理。

------



## 🧠 核心架构 (Core Architecture)



SSTNet 采用了一种新颖的**双流多模态架构 (Two-Stream Multimodal Architecture)** (v2.0 Global Aware 版)。

模型并行处理两条路径，最终融合以进行序列级预测：



### 1. 时序流 (Temporal Stream: The "How")



- **核心组件**: 基于 **Transformer Encoder**。
- **功能**: 捕捉注视行为的时序逻辑和演变过程（例如：先看了哪里，后看了哪里，停留了多久）。
- **多模态融合**: 该流不仅处理局部的视觉特征，还独特地融合了：
  - **全局视觉上下文 (Global Context)**: 来源于整张原图的 CLIP 特征，广播到每个时间步。
  - **生理特征 (Physiological Data)**: 包含注视点坐标 (X, Y)、注视时长和瞳孔直径。



### 2. 空间流 (Spatial Stream: The "What")



- **核心组件**: 基于 **NetVLAD (Vector of Locally Aggregated Descriptors)**。
- **功能**: 捕捉注视内容的空间分布统计信息，**忽略时间顺序**。
- **机制**: 通过聚类分析，统计受试者关注了多少次特定类型的视觉区域（例如，人脸区域 vs. 背景区域），生成一个描述内容分布的全局描述子。



### 3. 高级聚合 (Advanced Aggregation - Optional)



- **MIL 模型**: 提供了一个独立的 **多示例学习 (Multiple Instance Learning, MIL)** 模块 (`RobustGatedAttention`)。
- **功能**: 将同一个受试者观看多张图片产生的所有序列特征聚合为一个病人级的诊断结果，能够自动识别最具诊断价值的关键图片/序列。

------



## 🛠️ 环境准备 (Installation)



请确保您的环境满足以下要求：

- Python >= 3.8
- PyTorch (建议使用 GPU 版本)
- CUDA 环境（如果使用 GPU）



### 一键安装依赖



Bash

```
# 1. 安装通用依赖库
pip install pandas numpy tqdm scikit-learn matplotlib wandb openpyxl

# 2. 安装 OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# 3. (可选) 运行环境自检脚本确保一切正常
python check_env.py
```

------



## 📂 项目结构概览 (Project Structure)



```
SSTNet/
├── config.py                      # [核心配置] 所有的路径、超参数、模型参数定义
├── train.py                       # [主训练脚本] SSTNet 端到端训练与验证
├── train_mil.py                   # [MIL训练脚本] 第二阶段：病人级聚合训练
├── extract_features.py            # [工具脚本] 使用训练好的 SSTNet 提取特征供 MIL 使用
├── test.py                        # [测试脚本] 标准测试集推理
├── test_ems.py                    # [测试脚本] EMS 特定数据集推理
├── check_env.py                   # 环境自检工具
├── requirements.txt               # 依赖列表
├── README.md                      # 项目文档
│
├── models/                        # 模型定义目录
│   ├── sstnet.py                  # SSTNet 主模型 (组装双流)
│   ├── temporal_stream.py         # 时序流 (Transformer + 多模态融合)
│   ├── spatial_stream.py          # 空间流 (NetVLAD)
│   └── mil_model.py               # MIL 聚合模型 (Robust Gated Attention)
│
├── data_process/                  # 数据预处理流水线
│   ├── split_fix.py               # [Step 1] 清洗 Excel 并拆分为 TXT 序列
│   ├── generate_clip_features.py  # [Step 2] 提取 CLIP 视觉特征 (.npy)
│   ├── check_data_stats.py        # [分析] 统计生理数据分布，用于更新归一化参数
│   ├── check_fixation_stats.py    # [分析] 统计序列长度，用于设定 MAX_SEQ_LEN
│   └── inspect_npy.py             # [工具] 检查生成的特征文件内容
│
├── utils/                         # 通用工具箱
│   ├── dataloader.py              # PyTorch Dataset 和 DataLoader 实现
│   ├── loss.py                    # 损失函数定义 (如 BCEWithLogitsLoss)
│   ├── metrics.py                 # 评估指标计算 (AUC, ACC, F1 等)
│   └── misc.py                    # 杂项 (固定随机种子, 模型保存)
│
├── dataset/                       # 数据存放区 (需自行准备数据)
│   ├── Images/                    # 原始图片素材库
│   ├── Train_Valid.xlsx           # 训练/验证集划分表
│   ├── Train_Valid/Fixations/     # 训练集原始眼动 Excel
│   ├── Test/Fixations/            # 测试集原始眼动 Excel
│   ├── TXT/                       # (自动生成) 清洗后的 TXT 序列文件
│   └── output/                    # (自动生成) 提取的 CLIP 特征 (.npy)
│
└── checkpoints/                   # 模型权重保存目录
```

------



## 🚀 完整数据流水线 (Data Pipeline)



在开始训练之前，必须严格按照以下步骤处理数据。所有路径均在 `config.py` 中配置。



### 准备工作



将原始数据放置在正确的位置：

1. 图片文件 -> `dataset/Images/`
2. 训练集眼动 Excel -> `dataset/Train_Valid/Fixations/`
3. 测试集眼动 Excel -> `dataset/Test/Fixations/`
4. 划分表 -> `dataset/Train_Valid.xlsx`



### 步骤 1: 数据清洗与序列化



运行 `split_fix.py`。此脚本会读取原始 Excel，剔除越界坐标，并将数据按“受试者_图片”拆分为独立的 `.txt` 序列文件。

Bash

```
# 处理训练/验证集
python data_process/split_fix.py --train

# 处理测试集
python data_process/split_fix.py --test
```

*输出：生成的 `.txt` 文件将保存在 `dataset/TXT/` 目录下。*



### 步骤 2: 提取 CLIP 视觉特征



运行 `generate_clip_features.py`。此脚本模拟人眼观察，根据注视点坐标裁剪局部图像，并使用预训练的 CLIP 模型提取**局部特征 (Local)** 和**全局特征 (Global)**。

*注意：此步骤需要 GPU，且耗时较长。生成的 `.npy` 文件体积较大。*

Bash

```
# 提取训练/验证集特征
python data_process/generate_clip_features.py --train

# 提取测试集特征
python data_process/generate_clip_features.py --test
```

*输出：特征字典将保存为 `dataset/output/\*.npy` 文件。*



### 步骤 3 (可选): 数据分析与配置更新



为了获得最佳性能，建议运行统计脚本来校准 `config.py` 中的参数。

1. 运行 `python data_process/check_fixation_stats.py` 查看序列长度分布，确认 `MAX_SEQ_LEN` 设置是否合理。
2. 运行 `python data_process/check_data_stats.py` 获取生理数据的精确统计值（Min/Max），并更新 `config.py` 中的归一化边界参数。

------



## ⚡ 训练与评估 (Training & Evaluation)



项目支持标准的 K-Fold 交叉验证。训练过程会自动连接 WandB 进行可视化记录。



### 阶段一：SSTNet 端到端训练



直接训练 SSTNet 主模型。验证时采用 Mean Voting 策略聚合病人级别的预测结果。

Bash

```
# 训练第 0 折 (Fold 0)
python train.py --fold 0

# 训练其他折
python train.py --fold 1
# ...
```

*最佳模型将保存在 `checkpoints/best_model_foldX.pth`。*



### 阶段二 (高级)：多示例学习 (MIL) 训练



使用训练好的 SSTNet 作为特征提取器，训练一个专门的 MIL 聚合模型，以进一步提升病人级诊断性能。

1. 提取特征 Bag:

加载某一折最好的 SSTNet 模型，提取所有数据的高维特征。

Bash

```
# 使用 Fold 0 的模型提取特征
python extract_features.py --fold 0
```

2. 训练 MIL Aggregator:

基于提取的特征 Bag 训练 Robust Gated Attention 模型。

Bash

```
# 训练 MIL 模型 (使用 Fold 0 提取的特征)
python train_mil.py --fold 0 --lr 1e-4 --dropout 0.5
```

------



## ⚙️ 关键配置说明 (Configuration)



所有核心参数均在 `config.py` 中集中管理。修改配置后无需改动代码。

| **参数类别** | **参数名**         | **默认值** | **说明**                             |
| ------------ | ------------------ | ---------- | ------------------------------------ |
| **数据维度** | `INPUT_DIM`        | 512        | CLIP 原始特征维度                    |
|              | `PHYSIO_DIM`       | 4          | 生理特征维度 (X, Y, Duration, Pupil) |
|              | `MAX_SEQ_LEN`      | 32         | 序列最大长度 (截断/填充)             |
| **模型结构** | `HIDDEN_DIM`       | 128        | **核心瓶颈层维度**，控制模型容量     |
| **时序流**   | `TEMP_LAYERS`      | 2          | Transformer 层数                     |
|              | `TEMP_HEADS`       | 4          | 注意力头数 (需能被 HIDDEN_DIM 整除)  |
| **空间流**   | `SPATIAL_CLUSTERS` | 8          | NetVLAD 聚类中心数 (K)               |
| **训练参数** | `BATCH_SIZE`       | 64         | 批次大小                             |
|              | `LEARNING_RATE`    | 5e-4       | 初始学习率                           |
|              | `EPOCHS`           | 100        | 总训练轮数                           |

------



## ⚖️ License



[NONE]