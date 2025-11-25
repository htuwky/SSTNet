# SSTNet

### Data Preprocessing: `data_process/split_fix.py`

此脚本负责将原始的眼动数据从 Excel 格式转换为模型所需的序列化文本格式，并执行必要的数据清洗。

**主要功能 (Key Functions):**

1. **格式转换 (Format Conversion)**: 将以“受试者”为单位的 Excel (`.xlsx`) 文件拆解为以“受试者_图片”为单位的独立 `.txt` 文件，便于后续 DataLoader 读取。
2. **特征提取 (Feature Extraction)**: 从原始数据中提取 5 维关键特征：**注视点序号 (Index)**, **X坐标**, **Y坐标**, **注视时长 (Duration)**, **瞳孔大小 (Pupil)**。
3. **数据清洗 (Data Cleaning)**: 自动剔除超出屏幕分辨率（标准 1024x768）或坐标异常（负值）的无效注视点。

**输入与输出 (Input & Output):**

- **输入**: `dataset/Train_Valid/Fixations/*.xlsx` (包含原始眼动记录)
- **输出**: `dataset/Train_Valid/TXT/*.txt` (清洗后的序列文件，格式为 `Index,X,Y,Duration,Pupil`)

**使用方法 (Usage):**

Bash

```py
python data_process/split_fix.py --dataset_dir dataset
```

**check_fixation_stats.py**该文件统计每张图片的注释点数目，并最终决定采取32为最佳数目，所有注视点基本上都在32之内，并且32是8的倍数，gpu好处理，医学上讲究
宁缺毋滥，保护长尾数据（异常值）的完整性，远比节省一点点显存更重要，我们后期采取mask掩码，Transformer 的视角： 当 Transformer 计算注意力时，它会先检查 Mask。凡是 Mask 标记为 0 的位置，它会把注意力权重直接设为 负无穷。
结果：在数学运算上，这些填充点对真实特征的影响力也是 0。模型根本不会去“学习”这些填充数据，它只会专注于前面的真实注视点。
NetVLAD 的视角： 我们在代码里专门加了 masked_fill 逻辑。在聚类投票时，填充点的票数也被强制清零了。

generate_clip.py与rinet项目的不同是rinet项目处理整张图片的特征，但是我的脚本首先升级为clip，不是传统的特征提取器，其次我根据坐标提取特征，而rinet提取的为  
整张图片的特征，我提取的是x,y附近的特征，模拟人眼只看一部分，不看全局

inspect_npy.py脚本是检查生成的.npy文件，随机抽取检查的

config.py文件是全局的配置文件