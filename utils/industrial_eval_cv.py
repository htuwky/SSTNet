import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse
import sys

# 引入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.sstnet import SSTNet
from utils.dataloader import get_loader


def calculate_industrial_metrics(y_true, y_probs, threshold=0.5):
    """ 计算工业级核心指标 """
    y_pred = (y_probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "Threshold": threshold,
        "Sens": sensitivity,
        "Spec": specificity,
        "PPV": ppv,
        "NPV": npv
    }


def scan_thresholds(y_true, y_probs):
    """ 策略A: 阈值扫描 """
    print("\n📊 === Strategy A: Threshold Scanning (Full Dataset OOF) ===")
    print(f"{'Thres':<6} | {'Sens (漏诊率)':<12} | {'Spec (误诊率)':<12} | {'PPV':<8} | {'NPV':<8}")
    print("-" * 65)

    best_score = 0
    best_threshold = 0.5

    # 扫描 0.3 到 0.9
    for th in np.arange(0.3, 0.95, 0.05):
        m = calculate_industrial_metrics(y_true, y_probs, threshold=th)
        print(f"{th:.2f}   | {m['Sens']:.4f}       | {m['Spec']:.4f}       | {m['PPV']:.4f}   | {m['NPV']:.4f}")

        # 寻找甜点：假设 Sens 权重为 2，Spec 权重为 1
        # 硬性要求：Sensitivity 必须 > 0.85 (不能漏太多)
        score = 2 * m['Sens'] + m['Spec']
        if score > best_score and m['Sens'] > 0.85:
            best_score = score
            best_threshold = th

    print("-" * 65)
    print(f"💡 Recommended Threshold: {best_threshold:.2f}")
    return best_threshold


def evaluate_grey_zone(y_true, y_probs, low_th=0.3, high_th=0.7):
    """ 策略B: 灰区分析 """
    print(f"\n🚦 === Strategy B: Grey Zone Analysis [{low_th} - {high_th}] ===")

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    certain_mask = (y_probs < low_th) | (y_probs > high_th)
    uncertain_mask = ~certain_mask

    n_total = len(y_true)
    n_uncertain = np.sum(uncertain_mask)

    print(f"Total Samples: {n_total}")
    print(f"Gray Zone (Need Doctor): {n_uncertain} ({n_uncertain / n_total * 100:.1f}%)")
    print(f"Auto-Processed:        {n_total - n_uncertain}")

    if np.sum(certain_mask) > 0:
        y_pred_certain = (y_probs[certain_mask] > high_th).astype(int)
        y_true_certain = y_true[certain_mask]
        acc_certain = np.mean(y_pred_certain == y_true_certain)
        print(f"✅ Accuracy in Auto Zone: {acc_certain * 100:.2f}%")

        # 检查放行区是否有漏网之鱼
        missed = np.sum((y_true == 1) & (y_probs < low_th))
        print(f"⚠️ Critical Misses (Green Zone): {missed}")


def main():
    device = torch.device(config.DEVICE)

    # 容器：存放所有折的预测结果
    oof_labels = []
    oof_probs = []

    print("🚀 Starting 4-Fold Cross-Validation Evaluation (OOF)...")

    # 遍历 4 个折
    for fold_idx in range(4):
        print(f"\n🔄 Processing Fold {fold_idx}...")

        # 1. 构造模型路径
        ckpt_path = os.path.join(config.PROJECT_ROOT, 'checkpoints', f'best_model_fold{fold_idx}.pth')
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Warning: Checkpoint for Fold {fold_idx} not found! Skipping.")
            continue

        # 2. 加载模型
        model = SSTNet().to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # 3. 加载该折对应的验证集 (Val Set)
        # 注意：get_loader('val', fold_idx) 会自动返回这一折没参与训练的数据
        val_loader = get_loader('val', fold_idx=fold_idx, batch_size=config.BATCH_SIZE)

        # 4. 推理
        fold_probs = []
        fold_labels = []

        with torch.no_grad():
            for local_vis, global_vis, physio, mask, label, _ in tqdm(val_loader, desc=f"Fold {fold_idx}"):
                local_vis = local_vis.to(device)
                global_vis = global_vis.to(device).squeeze(1)
                physio = physio.to(device)
                mask = mask.to(device)

                logits = model(local_vis, global_vis, physio, mask)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                fold_probs.extend(probs)
                fold_labels.extend(label.numpy())

        # 收集结果
        oof_labels.extend(fold_labels)
        oof_probs.extend(fold_probs)

    # 5. 全局评估
    if len(oof_labels) == 0:
        print("❌ No data collected. Please check if checkpoints exist.")
        return

    print("\n" + "=" * 50)
    print(f"🌍 GLOBAL EVALUATION (Total Samples: {len(oof_labels)})")
    print("=" * 50)

    # 将列表转 numpy
    oof_labels = np.array(oof_labels)
    oof_probs = np.array(oof_probs)

    # 运行策略分析
    best_th = scan_thresholds(oof_labels, oof_probs)
    evaluate_grey_zone(oof_labels, oof_probs, low_th=0.3, high_th=0.7)


if __name__ == "__main__":
    main()