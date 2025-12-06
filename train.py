import os
import torch
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import sys
import argparse
import torch.nn as nn
# 导入项目模块
import config
from models.sstnet import SSTNet
from utils.dataloader import get_loader
from utils.loss import get_loss_function
from utils.metrics import calculate_metrics
from utils.misc import fix_seed, save_checkpoint
from utils.loss import LabelSmoothingBCEWithLogitsLoss,SupConLoss

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    训练一个 Epoch (v2.0 全局感知版)
    """
    model.train()
    running_loss = 0.0
    # [新增] 实例化对比 Loss
    criterion_supcon = SupConLoss(temperature=0.07).to(device)

    # [新增] 平衡系数 (分类占 1.0, 对比占 0.5)
    lambda_supcon = 0.5
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]")

    # [修改] 解包 6 个变量: local, global, physio, mask, label, id
    for step, (local_vis, global_vis, physio, mask, label, subject_ids) in enumerate(pbar):
        # # --- 新增调试代码 ---
        # if step == 0:
        #     print(f"\n🔍 [Debug Check] Labels in this batch: {label.tolist()}")
        #     print(f"🔍 [Debug Check] Subject IDs: {subject_ids}")
        # # ------------------
        #调试成功，删掉注释
        # # 1. 数据搬运
        local_vis = local_vis.to(device)  # [B, 32, 512]

        # [关键] 去掉中间的维度: [B, 1, 512] -> [B, 512]
        global_vis = global_vis.to(device).squeeze(1)

        physio = physio.to(device)
        mask = mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # [修改] 前向传播，要求返回投影特征
        # outputs 是 logits, proj_feats 是归一化的特征
        outputs, proj_feats = model(local_vis, global_vis, physio, mask, return_proj=True)

        # 1. 计算分类 Loss (主任务)
        loss_cls = criterion(outputs, label.float().unsqueeze(1))

        # 2. 计算对比 Loss (辅助任务)
        # 注意：SupCon 需要 label 来知道谁和谁是同类
        loss_con = criterion_supcon(proj_feats, label)

        # 3. 总 Loss
        loss = loss_cls + lambda_supcon * loss_con

        loss.backward()
        optimizer.step()

        # 6. 记录数据
        loss_val = loss.item()
        running_loss += loss_val

        # WandB 记录
        wandb.log({"train_batch_loss": loss_val})
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    验证模型 (病人级聚合评估)
    """
    model.eval()
    running_loss = 0.0

    # 结果容器: 按 subject_id 聚合
    patient_results = {}

    with torch.no_grad():
        # [修改] 解包 6 个变量
        for local_vis, global_vis, physio, mask, label, subject_ids in tqdm(loader, desc="[Val]"):
            local_vis = local_vis.to(device)
            global_vis = global_vis.to(device).squeeze(1)  # [B, 512]
            physio = physio.to(device)
            mask = mask.to(device)
            label = label.to(device)

            # 前向传播
            logits = model(local_vis, global_vis, physio, mask)

            # 计算 Loss
            loss = criterion(logits, label.float().unsqueeze(1))
            running_loss += loss.item()

            # 转概率
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_np = label.cpu().numpy()

            # 聚合逻辑
            for i, subj_id in enumerate(subject_ids):
                if subj_id not in patient_results:
                    patient_results[subj_id] = {'probs': [], 'label': labels_np[i]}
                patient_results[subj_id]['probs'].append(probs[i])

    # --- Patient-Level Metrics ---
    final_probs = []
    final_labels = []

    for subj_id, data in patient_results.items():
        # 策略: 平均分投票 (Mean Voting)
        avg_prob = np.mean(data['probs'])
        final_probs.append(avg_prob)
        final_labels.append(data['label'])

    metrics = calculate_metrics(final_labels, final_probs)
    metrics['loss'] = running_loss / len(loader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SSTNet v2.0 Training")
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Cross-validation fold index')
    args = parser.parse_args()

    current_fold = args.fold
    print(f"\n🚀 Starting Training for Fold: {current_fold} (v2.0 Global Aware)")
    print("=" * 40)

    fix_seed(config.SEED)

    wandb.init(
        project="SSTNet-v2",
        name=f"Fold{current_fold}_GlobalAware",
        config={
            "fold": current_fold,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "model": "SSTNet v2.0"
        }
    )

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loader = get_loader('train', fold_idx=current_fold)
    val_loader = get_loader('val', fold_idx=current_fold)

    # 构建模型
    model = SSTNet().to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    # 优化器 & 损失
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device), label_smoothing=0.1)
    # print("✅ Advanced Setup: BCEWithLogitsLoss with label_smoothing=0.1 enabled.")
    # [修改后]
    # 1. 定义正样本权重 (如果您的数据正负均衡，设为 1.0 即可)
    pos_weight = torch.tensor([1.0]).to(device)
    # 2. 实例化自定义的 Loss 类，设置平滑因子为 0.1
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.1, pos_weight=pos_weight)

    print("✅ Custom LabelSmoothingBCEWithLogitsLoss (smoothing=0.1) enabled.")
    #
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    best_auc = 0.0
    save_dir = os.path.join(config.PROJECT_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "val_auc": val_metrics['auc'],
            "val_acc": val_metrics['acc'],
            "val_f1": val_metrics['f1'],
            "val_sens": val_metrics['sensitivity'],
            "val_spec": val_metrics['specificity'],
            "val_prec": val_metrics['precision']
        })

        # print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        print(f"-" * 80)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {train_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"   Accuracy:    [{val_metrics['acc']:.4f}]")
        print(f"   Sensitivity: [{val_metrics['sensitivity']:.4f}]")
        print(f"   Specificity: [{val_metrics['specificity']:.4f}]")
        print(f"   AUC:         [{val_metrics['auc']:.4f}]")
        print(f"   Precision:   [{val_metrics['precision']:.4f}]")
        print(f"   F1-score:    [{val_metrics['f1']:.4f}]")
        print(f"-" * 80)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_path = os.path.join(save_dir, f"best_model_fold{current_fold}.pth")
            save_checkpoint(model, optimizer, epoch, best_auc, save_path)
            print(f"💾 Best Model Saved! AUC: {best_auc:.4f}")

    print(f"🏆 Fold {current_fold} Finished. Best AUC: {best_auc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()