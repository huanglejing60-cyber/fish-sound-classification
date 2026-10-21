# -*- coding: utf-8 -*-
"""入口 2：滑窗扫描与昼夜曲线。

单文件（30 分钟）：
    python scripts/02_scan.py --file "录音.wav"
全天目录（48 个 30 分钟文件，出 24 小时曲线）：
    python scripts/02_scan.py --dir "录音目录"
结果默认写入 outputs/。
"""
import os
import sys
import csv
import argparse
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from fishsound import config as C
from fishsound.infer import FishSoundDetector, detect_segments, parse_recording_timestamp

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def scan_one_file(det, path, out_dir):
    mid_sec, fish, proba = det.scan_file(path)
    hit = fish >= C.FISH_THRESH
    segs = detect_segments(mid_sec, fish)
    print(f"{os.path.basename(path)}：报鱼窗 {int(hit.sum())}/{len(mid_sec)}")
    for s, e, pk, spk in segs:
        print(f"  成片 {s:.1f}~{e:.1f} 分钟，峰值 {pk:.2f}")
    csv_path = os.path.join(out_dir, "单文件扫描结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["窗口", "中点秒", "分钟"] + det.labels)
        for i in range(len(mid_sec)):
            w.writerow([i, f"{mid_sec[i]:.2f}", f"{mid_sec[i]/60:.3f}"] +
                       [f"{v:.4f}" for v in proba[i]])
    plt.figure(figsize=(13, 5))
    m = mid_sec / 60
    plt.plot(m, fish, lw=1, alpha=.7, c="#2563eb")
    plt.scatter(m[hit], fish[hit], s=40, c="#dc2626", label="报鱼")
    plt.scatter(m[~hit], fish[~hit], s=12, c="#94a3b8", label="未报")
    plt.axhline(C.FISH_THRESH, ls="--", c="#dc2626")
    plt.ylim(0, 1); plt.xlabel("分钟"); plt.ylabel("鱼概率"); plt.legend()
    plt.title("单文件滑窗鱼概率")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "单文件鱼概率.png"), dpi=140)
    print("结果已写入", out_dir)


def scan_day(det, rec_dir, out_dir):
    files = sorted(os.path.join(rec_dir, f) for f in os.listdir(rec_dir)
                   if f.lower().endswith(".wav"))
    records, base = [], None
    for fp in files:
        ts = parse_recording_timestamp(os.path.basename(fp))
        if ts is None:
            continue
        if base is None:
            base = ts.replace(hour=0, minute=0, second=0)
        mid, fish, _ = det.scan_file(fp)
        segs = detect_segments(mid, fish)
        rel = (ts - base).total_seconds() / 3600
        seg_min = sum(e - s for s, e, _, _ in segs)
        records.append((rel, seg_min, float(fish.max()), ts.strftime("%H:%M")))
    csv_path = os.path.join(out_dir, "全天扫描结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["相对小时", "成片时长(分钟)", "鱼概率峰值", "时刻"])
        w.writerows(records)
    rel = np.array([r[0] for r in records]); seg = np.array([r[1] for r in records])
    plt.figure(figsize=(14, 5.5))
    plt.bar(rel, seg, width=.4, color="#1d4ed8")
    plt.xlabel("钟点"); plt.ylabel("成片发声时长（分钟）")
    xt = np.arange(int(rel.min()), int(rel.max()) + 2, 2)
    plt.xticks(xt, [f"{int(h % 24):02d}:00" for h in xt])
    plt.title("24 小时昼夜发声曲线")
    plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "昼夜曲线.png"), dpi=140)
    print("已出全天曲线，共", len(records), "个文件 ->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(C.MODEL_DIR, C.CLEAN_MODEL))
    ap.add_argument("--file", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    det = FishSoundDetector(args.model)
    print("模型加载完成（", det.backend, "）：", os.path.basename(args.model))

    if args.file:
        scan_one_file(det, args.file, out_dir)
    else:
        scan_day(det, args.dir or C.RECORDING_DIR, out_dir)


if __name__ == "__main__":
    main()
