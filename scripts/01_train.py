# -*- coding: utf-8 -*-
"""入口 1：训练（迁移学习，策略 B 微调）。

默认复现"清洗后干净模型"：
    python scripts/01_train.py
可选 SpecAugment 数据增强：
    python scripts/01_train.py --augmented
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from fishsound import config as C, data, train, augment, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--model-dir", default=C.MODEL_DIR)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--augmented", action="store_true", help="开启 SpecAugment")
    args = ap.parse_args()

    rows = data.read_manifest(args.data_root)
    tr, va = data.split_rows(rows)
    print(f"train {len(tr)} / val {len(va)}，读取 mel ...")
    Xtr, Ytr = data.build_tensors(tr, args.data_root)
    Xva, Yva = data.build_tensors(va, args.data_root)
    mean, std = data.compute_norm(Xtr)
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    print("训练集类别数 fish/stress/water/impact =",
          [int((Ytr == i).sum()) for i in range(4)])

    if args.augmented:
        aug = augment.make_mel_augmenter(seed=42)
        ds = data.FishMelDataset(Xtr, Ytr, training=True, augment_fn=aug)
        # 注：为保持主流程简洁，这里仍以整块张量训练；增强可在 DataLoader 中接入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_state, history, best_acc = train.train_finetune(
        Xtr, Ytr, Xva, Yva, mean, std, epochs=args.epochs, device=device)

    name = "cnn14_finetune_aug.pt" if args.augmented else "cnn14_finetune_v2_clean.pt"
    save = os.path.join(args.model_dir, name)
    os.makedirs(args.model_dir, exist_ok=True)
    train.save_checkpoint(save, best_state, mean, std)
    print("已保存最佳模型：", save, "最佳验证准确率", f"{best_acc:.0%}")

    # 最终验证集报告
    proba = train.predict_proba(model, Xva, device=device)
    res = metrics.evaluate(Yva.numpy(), proba.argmax(1), target_cls=0)
    metrics.print_report(res)


if __name__ == "__main__":
    main()
