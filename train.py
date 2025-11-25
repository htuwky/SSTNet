import os
import torch
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import sys
import argparse  # <--- [新增] 引入参数解析库

# 导入项目模块
import config
from models.sstnet import SSTNet
from utils.dataloader import get_loader
from utils.loss import get_loss_function
from utils.metrics import calculate_metrics
from utils.misc import fix_seed, save_checkpoint


# ... (train_one_epoch 和 validate 函数保持完全不变，这里省略以节省篇幅) ...
# 请保留原来的 train_one_epoch 和 validate 代码

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 Epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]")

    for step, (visual, physio, mask, label) in enumerate(pbar):
        visual = visual.to(device)
        physio = physio.to(device)
        mask = mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(visual, physio, mask)
        loss = criterion(outputs, label.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        wandb.log({"train_batch_loss": loss_val})
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for visual, physio, mask, label in tqdm(loader, desc="[Val]"):
            visual = visual.to(device)
            physio = physio.to(device)
            mask = mask.to(device)
            label = label.to(device)

            outputs = model(visual, physio, mask)
            loss = criterion(outputs, label.float().unsqueeze(1))

            running_loss += loss.item()
            all_logits.extend(outputs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    all_logits = np.concatenate(all_logits).flatten()
    all_labels = np.array(all_labels)

    metrics = calculate_metrics(all_labels, all_logits)
    metrics['loss'] = epoch_loss
    return metrics


def main():
    # --- [新增] 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="SSTNet Training")
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Cross-validation fold index (0-3)')
    args = parser.parse_args()

    current_fold = args.fold
    print(f"\n🚀 Starting Training for Fold: {current_fold}")
    print("=" * 30)

    # 1. 固定随机种子
    fix_seed(config.SEED)

    # 2. 初始化 WandB (名称动态化)
    wandb.init(
        project="SSTNet-Experiments",
        # [修改] 实验名称加上 Fold 后缀，方便区分
        name=f"Fold{current_fold}_SSTNet",
        config={
            "fold": current_fold,  # 记录当前的折数
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS,
            "model": "SSTNet",
            "loss": "FocalLoss"
        }
    )

    # 3. 环境设置
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 4. 数据加载 (使用命令行传入的 fold)
    # [修改] 这里的 fold_idx 变成动态变量
    train_loader = get_loader('train', fold_idx=current_fold)
    val_loader = get_loader('val', fold_idx=current_fold)

    # 5. 模型构建
    model = SSTNet().to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    # 6. 优化器与损失
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = get_loss_function()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    # 7. 训练主循环
    best_auc = 0.0
    save_dir = os.path.join(config.PROJECT_ROOT, 'checkpoints')

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_epoch_loss": train_loss,
            "val_auc": val_metrics['auc'],
            "val_acc": val_metrics['acc'],
            "val_f1": val_metrics['f1']
        })

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        # [修改] 保存模型时文件名加上 Fold 后缀
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_path = os.path.join(save_dir, f"best_model_fold{current_fold}.pth")
            save_checkpoint(model, optimizer, epoch, best_auc, save_path)
            print(f"💾 Best Model (Fold {current_fold}) Saved! AUC: {best_auc:.4f}")

    print(f"🏆 Fold {current_fold} Finished. Best AUC: {best_auc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()