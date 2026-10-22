# -*- coding: utf-8 -*-
"""入口 6：轻量模型端侧部署（fp32 ONNX -> 全 INT8 -> 混合 INT8）。

对 05 对比中训练好的 DS-CNN(0.14M)、MobileNet(1.5M)：
  1. 导出 fp32 ONNX（带预处理 sidecar）；
  2. 全 INT8 量化（所有层都量化，看会不会掉点）；
  3. 混合 INT8（首层 stem 卷积 + 分类头保留 fp32，敏感层不量化）；
  4. 统一测：磁盘体积、CPU 延迟（batch=1/16）、验证集准确率与真鱼 P/R/F1。
CNN14 直接复用第 19 课已生成的 fp32 / 混合 INT8 文件，做同口径横向对比。

产出：
  - results/12_轻量模型端侧部署.csv
  - results/12_轻量模型端侧部署.png
用法：
    python scripts/06_deploy_lightweight.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fishsound import config as C, data, deploy, metrics
from fishsound import models as M

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO, "results")

# 轻量模型：checkpoint 文件名 / 构建名 / 分类头 ONNX 节点名
LW = [
    dict(tag="DS-CNN", ck="dscnn.pt", name="dscnn", head="/model/fc/Gemm"),
    dict(tag="MobileNet", ck="mobilenet.pt", name="mobilenet", head="/model/fc2/Gemm"),
]
SENS_STEM = "/model/stem/stem.0/Conv"


def eval_onnx(path, Xva, Yva):
    """ONNX 推理验证集，返回 metrics 结果与预测标签。"""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = sess.run(None, {iname: Xva.numpy().astype(np.float32)})[0]
    pred = out.argmax(1)
    return metrics.evaluate(Yva.numpy(), pred, target_cls=0), pred


def mb(path):
    return os.path.getsize(path) / 1e6


def build_lightweight_rows(cfg, model_dir, Xtr_calib, Xva, Yva):
    """对一个轻量模型导出 fp32 / 全 INT8 / 混合 INT8 并测全部指标。"""
    ckpt = os.path.join(model_dir, cfg["ck"])
    fp32 = os.path.join(model_dir, f"{cfg['name']}_fp32.onnx")
    full = os.path.join(model_dir, f"{cfg['name']}_int8_full.onnx")
    mix = os.path.join(model_dir, f"{cfg['name']}_int8_mix.onnx")

    print(f"\n===== {cfg['tag']} 导出 fp32 ONNX =====", flush=True)
    deploy.export_onnx(ckpt, fp32, model_name=cfg["name"])
    max_err, ok = deploy.verify_onnx(ckpt, fp32, model_name=cfg["name"])
    print(f"[校验] PyTorch vs ONNX 最大误差 {max_err:.2e}（{'一致' if ok else '不一致!'}）", flush=True)

    print(f"[量化] 全 INT8 ...", flush=True)
    deploy.quantize_mixed_int8(fp32, Xtr_calib, full, sensitive_nodes=[])
    print(f"[量化] 混合 INT8（首层stem+分类头保留fp32）...", flush=True)
    deploy.quantize_mixed_int8(
        fp32, Xtr_calib, mix, sensitive_nodes=[SENS_STEM, cfg["head"]])

    rows = []
    for ver, path in [("fp32", fp32), ("INT8全量化", full), ("INT8混合", mix)]:
        res, _ = eval_onnx(path, Xva, Yva)
        b1, _ = deploy.benchmark(path, batch=1, n=40, warmup=5)
        b16, _ = deploy.benchmark(path, batch=16, n=30, warmup=5)
        row = dict(tag=cfg["tag"], ver=ver, size=mb(path), b1=b1, b16=b16,
                   acc=res["accuracy"], prec=res["precision"],
                   rec=res["recall"], f1=res["f1"])
        rows.append(row)
        print(f"[{cfg['tag']}/{ver}] {row['size']:.2f}MB b1={b1:.2f}ms "
              f"b16={b16:.2f}ms 准确率{res['accuracy']:.0%} 鱼F1{res['f1']:.0%}", flush=True)
    return rows


def build_cnn14_rows(model_dir, Xva, Yva):
    """CNN14 复用现成 fp32 / 混合 INT8 文件。"""
    fp32 = os.path.join(model_dir, C.ONNX_MODEL)
    mix = os.path.join(model_dir, C.INT8_MIXED_MODEL)
    rows = []
    for ver, path in [("fp32", fp32), ("INT8混合", mix)]:
        if not os.path.exists(path):
            print(f"[跳过] 找不到 {path}")
            continue
        res, _ = eval_onnx(path, Xva, Yva)
        b1, _ = deploy.benchmark(path, batch=1, n=30, warmup=5)
        b16, _ = deploy.benchmark(path, batch=16, n=20, warmup=3)
        row = dict(tag="CNN14", ver=ver, size=mb(path), b1=b1, b16=b16,
                   acc=res["accuracy"], prec=res["precision"],
                   rec=res["recall"], f1=res["f1"])
        rows.append(row)
        print(f"[CNN14/{ver}] {row['size']:.1f}MB b1={b1:.1f}ms "
              f"b16={b16:.1f}ms 准确率{res['accuracy']:.0%} 鱼F1{res['f1']:.0%}", flush=True)
    # 全 INT8 在第 19 课已验证会掉精度（首层/分类头被量化），此处不重复跑，
    # 仅以一行占位并标注，保证图表完整。
    rows.append(dict(tag="CNN14", ver="INT8全量化", size=np.nan, b1=np.nan, b16=np.nan,
                     acc=np.nan, prec=np.nan, rec=np.nan, f1=np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--model-dir", default=C.MODEL_DIR)
    args = ap.parse_args()

    rows_csv = data.read_manifest(args.data_root)
    tr, va = data.split_rows(rows_csv)
    print(f"train {len(tr)} / val {len(va)}，读取 mel ...")
    Xtr, Ytr = data.build_tensors(tr, args.data_root)
    Xva, Yva = data.build_tensors(va, args.data_root)
    ck = torch.load(os.path.join(args.model_dir, C.CLEAN_MODEL),
                    map_location="cpu", weights_only=False)
    mean, std = ck["mean"], ck["std"]
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    all_rows = []
    for cfg in LW:
        all_rows += build_lightweight_rows(cfg, args.model_dir, Xtr, Xva, Yva)
    all_rows += build_cnn14_rows(args.model_dir, Xva, Yva)

    save_csv(all_rows)
    save_figure(all_rows)
    print("\n完成。见 results/12_轻量模型端侧部署.csv / .png")


def save_csv(rows):
    os.makedirs(RESULT_DIR, exist_ok=True)
    import csv
    path = os.path.join(RESULT_DIR, "12_轻量模型端侧部署.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型", "版本", "大小(MB)", "CPU延迟b1(ms)", "CPU延迟b16(ms)",
                    "准确率", "真鱼精度", "召回", "F1"])
        for r in rows:
            def fmt(x, pct=False):
                if isinstance(x, float) and np.isnan(x):
                    return "掉精度(见19课)"
                return f"{x:.0%}" if pct else f"{x:.2f}"
            w.writerow([r["tag"], r["ver"], fmt(r["size"]), fmt(r["b1"]),
                        fmt(r["b16"]), fmt(r["acc"], 1), fmt(r["prec"], 1),
                        fmt(r["rec"], 1), fmt(r["f1"], 1)])
    print("已存表：", path)


def save_figure(rows):
    tags = ["DS-CNN", "MobileNet", "CNN14"]
    vers = ["fp32", "INT8全量化", "INT8混合"]
    vcolors = {"fp32": "#7f7f7f", "INT8全量化": "#d62728", "INT8混合": "#2ca02c"}

    def get(tag, ver, key):
        for r in rows:
            if r["tag"] == tag and r["ver"] == ver:
                return r[key]
        return np.nan

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(tags))
    w = 0.25

    # 体积
    ax = axes[0, 0]
    for i, ver in enumerate(vers):
        vals = [get(t, ver, "size") for t in tags]
        ax.bar(x + (i - 1) * w, vals, w, label=ver, color=vcolors[ver])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("模型体积（MB，对数轴）"); ax.set_title("磁盘体积")
    ax.legend()

    # b1 延迟
    ax = axes[0, 1]
    for i, ver in enumerate(vers):
        vals = [get(t, ver, "b1") for t in tags]
        ax.bar(x + (i - 1) * w, vals, w, label=ver, color=vcolors[ver])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("CPU batch=1 延迟（ms，对数轴）"); ax.set_title("单窗推理延迟")
    ax.legend()

    # b16 延迟
    ax = axes[1, 0]
    for i, ver in enumerate(vers):
        vals = [get(t, ver, "b16") for t in tags]
        ax.bar(x + (i - 1) * w, vals, w, label=ver, color=vcolors[ver])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("CPU batch=16 延迟（ms，对数轴）"); ax.set_title("批量推理延迟")
    ax.legend()

    # 鱼 F1（核心目标类）
    ax = axes[1, 1]
    for i, ver in enumerate(vers):
        vals = [get(t, ver, "f1") for t in tags]
        ax.bar(x + (i - 1) * w, vals, w, label=ver, color=vcolors[ver])
    ax.set_xticks(x); ax.set_xticklabels(tags); ax.set_ylim(0, 1.0)
    ax.set_ylabel("真鱼 F1"); ax.set_title("核心目标类（真鱼）F1 —— 量化是否伤性能")
    ax.legend()

    fig.suptitle("轻量模型端侧部署：fp32 / 全INT8 / 混合INT8 的体积·延迟·精度", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(RESULT_DIR, "12_轻量模型端侧部署.png")
    fig.savefig(path, dpi=130)
    print("已存图：", path)


if __name__ == "__main__":
    main()
