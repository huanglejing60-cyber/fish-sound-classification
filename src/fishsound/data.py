# -*- coding: utf-8 -*-
"""数据读取：manifest 划分、log-mel 张量组装、标准化统计、PyTorch Dataset。"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from .features import load_mel


def read_manifest(root=C.DATA_ROOT, name=C.MANIFEST):
    """读取 数据划分_manifest.csv（列：文件名,类别,类别id,源段,split,来源）。"""
    path = os.path.join(root, name)
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_rows(rows):
    return ([r for r in rows if r["split"] == "train"],
            [r for r in rows if r["split"] == "val"])


def build_tensors(rows, root=C.DATA_ROOT):
    """把若干 manifest 行读成 (mel 张量 N,64,300, 标签 N)。"""
    mels = np.stack([load_mel(os.path.join(root, r["类别"], r["文件名"])) for r in rows])
    labels = np.array([int(r["类别id"]) for r in rows], dtype=np.int64)
    return torch.from_numpy(mels).float(), torch.from_numpy(labels).long()


def compute_norm(X_train):
    """训练集全局均值/标准差，验证/推理共用同一套。"""
    return float(X_train.mean()), float(X_train.std() + 1e-6)


class FishMelDataset(Dataset):
    """供 DataLoader 使用：训练时可挂数据增强。"""

    def __init__(self, X, y, training=False, augment_fn=None):
        self.X = X
        self.y = y
        self.training = training
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.training and self.augment_fn is not None:
            x = self.augment_fn(x)
        return x, self.y[i]
