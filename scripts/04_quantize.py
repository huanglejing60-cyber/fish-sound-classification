# -*- coding: utf-8 -*-
"""入口 4：静态 INT8 混合精度量化，并对比体积/延迟/准确率。

    python scripts/04_quantize.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
import onnxruntime as ort

from fishsound import config as C, data, deploy, metrics


def onnx_eval(path, X, y, batch=16):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = []
    for s in range(0, len(X), batch):
        out.append(sess.run(["logits"], {"mel": X[s:s + batch]})[0])
    proba = _softmax(np.concatenate(out))
    return metrics.evaluate(y, proba.argmax(1), target_cls=0)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=C.DATA_ROOT)
    ap.add_argument("--model-dir", default=C.MODEL_DIR)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fp32", default=None)
    ap.add_argument("--int8", default=None)
    ap.add_argument("--n-calib", type=int, default=122)
    args = ap.parse_args()

    import time
    t0 = time.time()
    rows = data.read_manifest(args.data_root)
    tr, va = data.split_rows(rows)
    print(f"读取 {len(tr)} 训练 + {len(va)} 验证 mel ...", flush=True)
    Xtr, Ytr = data.build_tensors(tr, args.data_root)
    Xva, Yva = data.build_tensors(va, args.data_root)
    print(f"读数据完成，耗时 {time.time()-t0:.0f}s", flush=True)

    ckpt = args.ckpt or os.path.join(args.model_dir, C.CLEAN_MODEL)
    fp32 = args.fp32 or os.path.join(args.model_dir, C.ONNX_MODEL)
    int8 = args.int8 or os.path.join(args.model_dir, C.INT8_MIXED_MODEL)

    # 标准化统计必须用训练时存进 checkpoint 的 mean/std，量化校准分布才与
    # 模型期望输入一致；checkpoint 里没有时才退化为现算。
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "mean" in ck and "std" in ck:
        mean, std = float(ck["mean"]), float(ck["std"])
        print(f"用 checkpoint 标准化: mean={mean:.4f} std={std:.4f}", flush=True)
    else:
        mean, std = data.compute_norm(Xtr)
        print("checkpoint 无 mean/std，用训练集现算", flush=True)
    calib = ((Xtr.numpy() - mean) / std)[:args.n_calib].astype(np.float32)
    Xva_n = ((Xva.numpy() - mean) / std).astype(np.float32)
    yva = Yva.numpy()

    if not os.path.exists(fp32):
        deploy.export_onnx(ckpt, fp32)

    print(f"量化中（{args.n_calib} 校准样本，Entropy，混合精度）...", flush=True)
    t1 = time.time()
    deploy.quantize_mixed_int8(fp32, calib, int8, method="entropy")
    print(f"量化完成，耗时 {time.time()-t1:.0f}s", flush=True)

    res_f = onnx_eval(fp32, Xva_n, yva)
    res_i = onnx_eval(int8, Xva_n, yva)
    print("延迟基准（batch=16，各80次）...", flush=True)
    t2 = time.time()
    lat_f = deploy.benchmark(fp32, batch=16)
    lat_i = deploy.benchmark(int8, batch=16)
    print(f"基准完成，耗时 {time.time()-t2:.0f}s", flush=True)

    def mb(p):
        return os.path.getsize(p) / 1024 / 1024

    print("\n================ 部署对比 ================")
    print(f"{'':10}{'fp32':>14}{'INT8混合':>14}")
    print(f"{'体积MB':10}{mb(fp32):>14.1f}{mb(int8):>14.1f}")
    print(f"{'b16延迟ms':10}{lat_f[0]:>14.1f}{lat_i[0]:>14.1f}")
    print(f"{'准确率':10}{res_f['accuracy']:>14.0%}{res_i['accuracy']:>14.0%}")
    print(f"{'真鱼精度':10}{res_f['precision']:>14.0%}{res_i['precision']:>14.0%}")
    print(f"{'真鱼召回':10}{res_f['recall']:>14.0%}{res_i['recall']:>14.0%}")
    print("体积压缩", f"{mb(fp32)/mb(int8):.1f}x；",
          "加速", f"{lat_f[0]/lat_i[0]:.1f}x")


if __name__ == "__main__":
    main()
