import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(y_true, y_pred_probs): # 修改参数名为 probs 以示区分
    """
    计算二分类指标
    Args:
        y_true: 真实标签
        y_pred_probs: 预测概率 (0~1之间)，不要传 Logits！
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # [删除] 不要在这里做 Sigmoid，因为输入已经是概率了
    # y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))

    # 1. 将概率转为 0/1 预测结果 (阈值 0.5)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # 2. 计算指标
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5

    try:
        # 添加 labels=[0, 1] 保证矩阵形状正确
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
        "sensitivity": sensitivity,
        "specificity": specificity
    }

    return metrics