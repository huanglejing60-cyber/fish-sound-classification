# -*- coding: utf-8 -*-
"""评估指标：混淆矩阵 / 准确率 / 精度 / 召回 / F1。

记忆口诀：
- 精度 Precision = 报上来的里有多少真的（宁可误报少，导师红线）
- 召回 Recall    = 真的里抓到了多少（宁可漏报少）
- 本项目按导师要求"精度优先、宁漏勿错"。
"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def class_prf(cm, cls=0):
    """返回某一类（默认 fish=0）的 precision / recall / f1。"""
    tp = cm[cls, cls]
    fp = cm[:, cls].sum() - tp
    fn = cm[cls, :].sum() - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def evaluate(y_true, y_pred, n_classes=4, target_cls=0):
    """一次性返回准确率 + 目标类 P/R/F1 + 混淆矩阵。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f1 = class_prf(cm, target_cls)
    return {"accuracy": accuracy(y_true, y_pred),
            "precision": p, "recall": r, "f1": f1, "cm": cm}


def print_report(result, labels=None):
    labels = labels or ["fish", "stress", "water", "impact"]
    cm = result["cm"]
    print("【混淆矩阵】行=真实，列=预测")
    print("        " + " ".join(f"{l:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>7}  " + " ".join(f"{cm[i, j]:>7}" for j in range(cm.shape[1])))
    print(f"准确率 {result['accuracy']:.0%} | 真鱼精度 {result['precision']:.0%} "
          f"| 召回 {result['recall']:.0%} | F1 {result['f1']:.0%}")


def C_LABELS():
    return ["fish", "stress", "water", "impact"]
