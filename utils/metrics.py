import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred_logits):
    """
    计算二分类的所有关键指标
    Args:
        y_true: 真实标签列表 (List or Numpy)
        y_pred_logits: 模型输出的 Logits 列表 (List or Numpy)
    Returns:
        dict: 包含 auc, acc, f1, pre, rec
    """
    # 转为 Numpy 数组
    y_true = np.array(y_true)
    y_pred_logits = np.array(y_pred_logits)

    # 1. 将 Logits 转为概率 (Sigmoid)
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))

    # 2. 将概率转为 0/1 预测结果 (阈值 0.5)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # 3. 计算指标
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        # 如果 Batch 里只有一个类别 (全0或全1)，AUC 没法算
        auc = 0.5

    metrics = {
        "auc": auc,
        "acc": accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0)
    }

    return metrics