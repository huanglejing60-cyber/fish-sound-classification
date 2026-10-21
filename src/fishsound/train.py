# -*- coding: utf-8 -*-
"""训练：CNN14 迁移学习（策略 B，解冻最后两块卷积 + 全连接头）。

- 先冻结全部参数，再放开 conv_block5/6、fc1、分类头；
- 冻结的浅层卷积块里 BN 始终 eval（不更新滑动统计）；
- 类别不平衡 -> 加权交叉熵；
- 小显存 -> 小 batch + 梯度累积；混合精度 AMP 加速；
- 按验证准确率保存最佳权重。
"""
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .models import build_pretrained_cnn14

FROZEN_BLOCKS = ["conv_block1", "conv_block2", "conv_block3", "conv_block4", "bn0"]


def class_weights(y, n_classes=len(C.LABELS)):
    cnt = np.array([(y.cpu() == i).sum().item() for i in range(n_classes)])
    return len(y) / (n_classes * np.maximum(cnt, 1))


def set_train_mode(model, frozen=FROZEN_BLOCKS):
    model.train()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d) and name.split(".")[0] in frozen:
            mod.eval()


def train_finetune(Xtr, Ytr, Xva, Yva, mean, std,
                   epochs=20, batch=4, accum=2, lr=1e-4, weight_decay=1e-4,
                   device=None, log_every=1, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model = build_pretrained_cnn14()
    for p in model.parameters():
        p.requires_grad = False
    for i in (5, 6):
        for p in getattr(model, f"conv_block{i}").parameters():
            p.requires_grad = True
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.fc_audioset.parameters():
        p.requires_grad = True
    model.to(device)

    w = torch.tensor(class_weights(Ytr), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)

    history = {"acc": [], "precision": [], "recall": [], "loss": []}
    best_acc, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        set_train_mode(model)
        idx = torch.randperm(len(Xtr))
        opt.zero_grad()
        total = 0.0
        for bi0 in range(0, len(Xtr), batch):
            bi = idx[bi0:bi0 + batch]
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out, _ = model(Xtr[bi])
                loss = criterion(out, Ytr[bi]) / accum
            scaler.scale(loss).backward()
            total += loss.item() * accum
            if (bi0 // batch + 1) % accum == 0 or (bi0 + batch) >= len(Xtr):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        acc, prec, rec = evaluate_model(model, Xva, Yva, batch, device)
        history["acc"].append(acc)
        history["precision"].append(prec)
        history["recall"].append(rec)
        history["loss"].append(total / len(Xtr) * batch)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"{ep:02d}轮 loss{history['loss'][-1]:.3f} 验证{acc:.0%} "
                  f"鱼精度{prec:.0%} 鱼召回{rec:.0%}", flush=True)

    return model, best_state, history, best_acc


@torch.no_grad()
def predict_proba(model, X, batch=16, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    outs = []
    for s in range(0, len(X), batch):
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(X[s:s + batch].to(device))
        outs.append(torch.softmax(logits, 1).float().cpu())
    return torch.cat(outs).numpy()


def evaluate_model(model, Xva, Yva, batch=16, device=None):
    from .metrics import evaluate
    proba = predict_proba(model, Xva, batch, device)
    res = evaluate(Yva.cpu().numpy(), proba.argmax(1), target_cls=0)
    return res["accuracy"], res["precision"], res["recall"]


def save_checkpoint(path, state_dict, mean, std, labels=C.LABELS):
    torch.save({"state_dict": state_dict, "mean": float(mean),
                "std": float(std), "labels": labels}, path)
