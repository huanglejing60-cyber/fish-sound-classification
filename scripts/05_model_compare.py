# -*- coding: utf-8 -*-
"""入口 5：轻量模型选型对比（DS-CNN / MobileNet / CNN14）。

在**完全相同的数据、log-mel 特征、标准化、训练超参、评估集**下，对比三档模型：
  - DS-CNN    约 0.14M，从零训练（ARM 关键词唤醒风格，极致端侧）
  - MobileNet 约 1.5M，从零训练（MobileNet 风格深度可分离卷积）
  - CNN14     约 80M，AudioSet 预训练 + 迁移学习（加载现成 best checkpoint）

对每个模型记录：参数量、fp32 理论大小、验证准确率、真鱼精度/召回/F1、
CPU 上 batch=1 / batch=16 推理延迟。产出：
  - results/11_轻量模型对比.csv
  - results/11_轻量模型对比.png
两个轻量模型的 best 权重也会存到模型目录，便于后续做 ONNX/INT8 部署对比。

用法：
    python scripts/05_model_compare.py
    python scripts/05_model_compare.py --epochs 40
"""
import os
import sys
import time
import copy
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fishsound import config as C, data, models, train, metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO, "results")


def load_cnn14_checkpoint(model_dir, device):
    """加载现成的清洗后 CNN14 best checkpoint，返回模型与统一标准化参数。"""
    path = os.path.join(model_dir, C.CLEAN_MODEL)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = models.build_pretrained_cnn14(classes=len(C.LABELS))
    model.load_state_dict(ck["state_dict"])
    return model, ck["mean"], ck["std"]


@torch.no_grad()
def cpu_latency(model, X, batch=1, n_warm=5, n_run=30):
    """CPU 上 batch=B 的单批推理延迟（毫秒，取中位数）。"""
    m = copy.deepcopy(model).cpu().eval()
    x = X[:batch].cpu()
    for _ in range(n_warm):
        m(x)
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def evaluate_on(model, Xva, Yva, device):
    proba = train.predict_proba(model, Xva, batch=16, device=device)
    res = metrics.evaluate(Yva.numpy(), proba.argmax(1), target_cls=0)
    return res


def collect(name, model, best_state, Xva, Yva, device, mean, std):
    """载入 best 权重 -> 评估指标 -> 统计参数量/体积 -> CPU 延迟。"""
    model.load_state_dict(best_state)
    model.to(device)
    res = evaluate_on(model, Xva, Yva, device)
    n_params = models.count_params(model)
    print(f"[{name}] 准确率{res['accuracy']:.0%} 真鱼精度{res['precision']:.0%} "
          f"召回{res['recall']:.0%} F1{res['f1']:.0%} 参数{n_params/1e6:.3f}M", flush=True)
    lat1 = cpu_latency(model, Xva, batch=1)
    lat16 = cpu_latency(model, Xva, batch=16)
    print(f"[{name}] CPU延迟 batch1={lat1:.1f}ms batch16={lat16:.1f}ms", flush=True)
    return {"name": name, "params_M": n_params / 1e6, "size_MB": n_params * 4 / 1e6,
            "accuracy": res["accuracy"], "precision": res["precision"],
            "recall": res["recall"], "f1": res["f1"],
            "lat_b1_ms": lat1, "lat_b16_ms": lat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--model-dir", default=C.MODEL_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)

    rows = data.read_manifest(args.data_root)
    tr, va = data.split_rows(rows)
    print(f"train {len(tr)} / val {len(va)}，读取 mel ...")
    Xtr, Ytr = data.build_tensors(tr, args.data_root)
    Xva, Yva = data.build_tensors(va, args.data_root)

    # 统一用 CNN14 训练时保存的训练集 mean/std（与 19 课同口径）；
    # 它是数据的标准化统计量，三个模型应共用，保证输入一致、CNN14 基线不偏移。
    cnn14, mean, std = load_cnn14_checkpoint(args.model_dir, device)
    now_mean, now_std = data.compute_norm(Xtr)
    print(f"标准化参数：checkpoint mean={mean:.4f} std={std:.4f} | "
          f"当前数据现算 mean={now_mean:.4f} std={now_std:.4f}")
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    results = []

    # 1) CNN14（预训练迁移，加载现成 best，不重训）
    results.append(collect("CNN14(80M,预训练)", cnn14, cnn14.state_dict(),
                           Xva, Yva, device, mean, std))
    del cnn14
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # 2) DS-CNN（从零）
    print("\n===== 从零训练 DS-CNN =====", flush=True)
    dscnn = models.build_lightweight("dscnn")
    _, ds_state, _, ds_acc = train.train_from_scratch(
        Xtr, Ytr, Xva, Yva, dscnn, epochs=args.epochs, device=device)
    train.save_checkpoint(os.path.join(args.model_dir, "dscnn.pt"),
                          ds_state, mean, std)
    results.append(collect("DS-CNN(0.14M)", models.build_lightweight("dscnn"),
                           ds_state, Xva, Yva, device, mean, std))

    # 3) MobileNet（从零）
    print("\n===== 从零训练 MobileNet =====", flush=True)
    mob = models.build_lightweight("mobilenet")
    _, mb_state, _, mb_acc = train.train_from_scratch(
        Xtr, Ytr, Xva, Yva, mob, epochs=args.epochs, device=device)
    train.save_checkpoint(os.path.join(args.model_dir, "mobilenet.pt"),
                          mb_state, mean, std)
    results.append(collect("MobileNet(1.5M)", models.build_lightweight("mobilenet"),
                           mb_state, Xva, Yva, device, mean, std))

    # 排序：轻量在前，重型在后
    order = ["DS-CNN(0.14M)", "MobileNet(1.5M)", "CNN14(80M,预训练)"]
    results.sort(key=lambda r: order.index(r["name"]))
    save_table_and_figure(results)
    print("\n完成。结果见 results/11_轻量模型对比.csv / .png")


def save_table_and_figure(results):
    os.makedirs(RESULT_DIR, exist_ok=True)
    import csv
    csv_path = os.path.join(RESULT_DIR, "11_轻量模型对比.csv")
    cols = ["name", "params_M", "size_MB", "accuracy", "precision",
            "recall", "f1", "lat_b1_ms", "lat_b16_ms"]
    cn = ["模型", "参数量(M)", "fp32大小(MB)", "准确率", "真鱼精度",
          "召回", "F1", "CPU延迟b1(ms)", "CPU延迟b16(ms)"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cn)
        for r in results:
            w.writerow([r["name"], f"{r['params_M']:.3f}", f"{r['size_MB']:.2f}",
                        f"{r['accuracy']:.1%}", f"{r['precision']:.1%}",
                        f"{r['recall']:.1%}", f"{r['f1']:.1%}",
                        f"{r['lat_b1_ms']:.1f}", f"{r['lat_b16_ms']:.1f}"])
    print("已存表：", csv_path)

    names = [r["name"] for r in results]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 参数量
    ax = axes[0, 0]
    bars = ax.bar(names, [r["params_M"] for r in results], color=colors)
    ax.set_yscale("log"); ax.set_ylabel("参数量（M，对数轴）")
    ax.set_title("参数量（越小越适合端侧）")
    for b, r in zip(bars, results):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{r['params_M']:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelrotation=12)

    # CPU 延迟 batch1
    ax = axes[0, 1]
    bars = ax.bar(names, [r["lat_b1_ms"] for r in results], color=colors)
    ax.set_yscale("log"); ax.set_ylabel("CPU batch=1 延迟（ms，对数轴）")
    ax.set_title("单窗 CPU 推理延迟（端侧实时性）")
    for b, r in zip(bars, results):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{r['lat_b1_ms']:.1f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelrotation=12)

    # 指标分组柱状
    ax = axes[1, 0]
    x = np.arange(len(names)); w = 0.25
    for i, (key, label) in enumerate(zip(["accuracy", "precision", "recall"],
                                         ["准确率", "真鱼精度", "召回"])):
        ax.bar(x + (i - 1) * w, [r[key] for r in results], w, label=label)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.08); ax.set_ylabel("分数"); ax.legend()
    ax.set_title("验证集表现（val=40）")

    # 帕累托：参数量 vs 准确率，点大小=延迟
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        ax.scatter(r["params_M"], r["accuracy"], s=max(60, 600 / r["lat_b1_ms"] * 8),
                   c=c, alpha=0.7, edgecolors="black", zorder=3)
        ax.annotate(r["name"], (r["params_M"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_xlabel("参数量（M，对数轴）"); ax.set_ylabel("验证准确率")
    ax.set_title("选型权衡：精度 vs 体积（点越大=延迟越低）")
    ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle("轻量模型选型对比：DS-CNN / MobileNet / CNN14（同数据同口径）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = os.path.join(RESULT_DIR, "11_轻量模型对比.png")
    fig.savefig(png_path, dpi=130)
    print("已存图：", png_path)


if __name__ == "__main__":
    main()
