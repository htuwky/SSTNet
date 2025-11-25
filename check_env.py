import torch
import clip
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 测试 CLIP
try:
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    print("✅ CLIP loaded successfully!")
except Exception as e:
    print(f"❌ CLIP Error: {e}")

print("✅ Environment is ready for SSTNet!")