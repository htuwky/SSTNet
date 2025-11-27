import os
import sys
import platform
import importlib.util
from pkg_resources import get_distribution, DistributionNotFound
import torch  # 提前导入 torch 以便检查 CUDA

# --- 0. 设置项目路径以便导入 config ---
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中，以便可以导入同目录下的模块
sys.path.append(current_dir)
try:
    import config

    config_loaded = True
    print(f"✅ Successfully loaded configuration from: {config.__file__}")
except ImportError:
    config_loaded = False
    print("❌ Warning: Could not load 'config.py'. Path checks will be skipped.")
    print(f"   Current directory check: {current_dir}")


def print_header(title):
    """打印带有格式的标题栏"""
    print(f"\n{'=' * 80}\n🔍 {title.upper()}\n{'=' * 80}")


def check_package(package_name, import_name=None):
    """
    检查指定包是否安装，并打印其版本号。

    Args:
        package_name: pip安装时的包名 (如 scikit-learn)
        import_name:代码导入时的模块名 (如 sklearn)。如果未提供，默认与 package_name 相同。
    """
    if import_name is None:
        import_name = package_name

    try:
        # 1. 首先检查模块是否可以被导入
        if importlib.util.find_spec(import_name) is None:
            print(f"   ❌ Missing Library: {package_name}")
            return False

        # 2. 尝试获取包的版本信息
        try:
            # 优先使用 pkg_resources 获取 pip 安装的版本号，这通常是最准确的
            version = get_distribution(package_name).version
        except DistributionNotFound:
            # 如果找不到（例如是内置库或通过特殊方式安装的），尝试从模块本身的属性中获取
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Version not found')
            except:
                version = 'Installed (Version unknown)'

        # 打印格式化的包名和版本号
        print(f"   ✅ {package_name:<25} : {version}")
        return True
    except ImportError:
        print(f"   ❌ Error importing: {package_name} (Module name: {import_name})")
        return False


# ==========================================
# 主检查流程
# ==========================================
if __name__ == "__main__":
    all_checks_passed = True

    # --- 1. 系统与 Python 环境 ---
    print_header("1. System & Python Information")
    print(f"   OS Platform     : {platform.platform()}")
    print(f"   Python Exec     : {sys.executable}")
    # 获取 Python 版本号的第一行
    # 先在外面处理好字符串
    python_version_str = sys.version.split('\n')[0]
    # 然后再放入 f-string
    print(f"   Python Version  : {python_version_str}")
    # print(f"   Python Version  : {sys.version.split('\n')[0]}")

    # --- 2. PyTorch & CUDA 深度诊断 ---
    print_header("2. PyTorch & Hardware Acceleration")
    print(f"   PyTorch Version : {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available  : {'✅ Yes' if cuda_available else '❌ No'}")

    if cuda_available:
        try:
            # 打印 CUDA 和 cuDNN 的详细版本信息
            print(f"   CUDA Version    : {torch.version.cuda}")
            print(f"   cuDNN Version   : {torch.backends.cudnn.version()}")
            device_count = torch.cuda.device_count()
            print(f"   GPU Device Count: {device_count}")
            # 遍历并打印每个 GPU 的名称和计算能力
            for i in range(device_count):
                print(
                    f"     Logs GPU {i}: {torch.cuda.get_device_name(i)} (Capability: {torch.cuda.get_device_capability(i)})")
        except Exception as e:
            print(f"   ⚠️ Error getting CUDA details: {e}")
    else:
        print("   ⚠️ Running on CPU. This will be significantly slower for training.")

    # --- 3. 核心依赖库检查 (基于您的环境列表) ---
    print_header("3. Key Library Versions Checks")
    # 格式: (pip安装包名, import模块名)
    # 这里列出了您环境中最重要的科学计算和深度学习库
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("pillow", "PIL"),  # Pillow 导入名是 PIL
        ("opencv-python", "cv2"),  # OpenCV 导入名是 cv2
        ("tqdm", "tqdm"),
        ("wandb", "wandb"),
        ("openpyxl", "openpyxl"),  # 用于 pandas 读取 Excel
        ("huggingface_hub", "huggingface_hub"),
        ("timm", "timm"),
        # CLIP 是通过 git 安装的，pkg_resources 可能找不到标准版本号，但可以检查是否安装
        ("clip", "clip"),
    ]

    print(f"   {'Package Name':<25} : {'Version'}")
    print("   " + "-" * 50)
    for pkg_name, import_name in required_packages:
        if not check_package(pkg_name, import_name):
            all_checks_passed = False

    # --- 4. 项目路径与配置检查 (基于 config.py) ---
    if config_loaded:
        print_header("4. Project Path Verification (from config.py)")
        # 定义需要检查的关键路径变量名及其对应的值
        paths_to_check = [
            ("PROJECT_ROOT", config.PROJECT_ROOT),
            ("DATASET_DIR", config.DATASET_DIR),
            ("IMAGE_DIR", config.IMAGE_DIR),
            ("TRAIN_FIXATIONS_DIR", config.TRAIN_FIXATIONS_DIR),
            ("TEST_FIXATIONS_DIR", config.TEST_FIXATIONS_DIR),
            # 输出目录如果不存在可以警告，不一定是错误，因为代码可能会自动创建
            ("OUTPUT_DIR", config.OUTPUT_DIR),
        ]

        for name, path in paths_to_check:
            # 检查路径是否存在
            exists = os.path.exists(path)
            status = "✅ Found" if exists else "❌ Not Found"

            # 对于某些还没生成的输出目录，给予黄色警告而不是红色错误
            if not exists and ("OUTPUT" in name or "TXT" in name):
                status = "⚠️ Not yet created (OK)"
            elif not exists:
                # 关键输入目录不存在则标记为失败
                all_checks_passed = False

            print(f"   {status:<20} | {name:<25} : {path}")

    # --- 5. 模型加载测试 (CLIP) ---
    print_header("5. Model Loading Test (CLIP)")
    clip_loaded = False
    try:
        import clip

        # 确定运行设备：如果配置加载成功则用配置的，否则有 GPU 用 GPU，没有用 CPU
        if config_loaded:
            target_device = config.DEVICE
        else:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # 确定模型名称
        model_name = config.CLIP_MODEL_NAME if config_loaded else "ViT-B/32"

        print(f"   Attempting to load CLIP '{model_name}' on device: [{target_device}]...")

        # 加载模型和预处理转换
        model, preprocess = clip.load(model_name, device=target_device)

        # 创建一个随机的虚拟图像张量进行推理测试
        # 形状为 [Batch=1, Channels=3, Height=224, Width=224]
        dummy_input = torch.randn(1, 3, 224, 224).to(target_device)

        # 在不计算梯度的上下文中执行前向传播
        with torch.no_grad():
            model.encode_image(dummy_input)

        print(f"   ✅ CLIP model loaded and basic inference test passed on {target_device}!")
        clip_loaded = True
    except ImportError:
        print("   ❌ Error: CLIP library (`clip`) not found.")
        all_checks_passed = False
    except Exception as e:
        # 捕获加载或推理过程中的其他错误（如显存不足、模型文件损坏等）
        print(f"   ❌ Error loading/running CLIP: {e}")
        print("      (Hint: Check internet connection for first download, or CUDA memory if on GPU)")
        all_checks_passed = False

    # --- 6. 最终总结 ---
    print("\n" + "=" * 80)
    # 只有当所有关键包都存在、配置加载成功且 CLIP 模型测试通过时，才认为环境就绪
    if all_checks_passed and config_loaded and clip_loaded:
        print("🚀 READY TO LAUNCH! Environment configuration looks good for SSTNet.")
    else:
        print("⚠️ ENVIRONMENT CHECKS FAILED. Please review the ❌ marks above to fix issues.")
    print("=" * 80 + "\n")