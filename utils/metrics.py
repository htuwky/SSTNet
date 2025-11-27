import numpy as np # 移除 torch，只保留 numpy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(y_true, y_pred_logits):
    """
    计算二分类的所有关键指标 (含 Specificity)
    """
    y_true = np.array(y_true)
    y_pred_logits = np.array(y_pred_logits)

    # 1. Logits -> Probabilities
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))

    # 2. Probabilities -> Binary Labels (Threshold 0.5)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # 3. 计算基础指标
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5

    # 4. 计算混淆矩阵 (Sensitivity & Specificity)
    # [优化] 添加 labels=[0, 1] 确保始终返回 2x2 矩阵，即使当前 batch 缺少某类样本
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    except ValueError:
        specificity = 0.0
        sensitivity = 0.0

    metrics = {
        "auc": auc,
        "acc": accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "sensitivity": sensitivity, # 即 Recall
        "specificity": specificity
    }

    return metrics