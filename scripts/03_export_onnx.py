# -*- coding: utf-8 -*-
"""入口 3：把训练好的 PyTorch 模型导出 ONNX 并做数值校验。

    python scripts/03_export_onnx.py
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fishsound import config as C, deploy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(C.MODEL_DIR, C.CLEAN_MODEL))
    ap.add_argument("--onnx", default=os.path.join(C.MODEL_DIR, C.ONNX_MODEL))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    print("导出 ONNX ...")
    deploy.export_onnx(args.ckpt, args.onnx)
    size_mb = os.path.getsize(args.onnx) / 1024 / 1024
    print(f"已导出：{args.onnx}（{size_mb:.1f} MB）")

    max_err, ok = deploy.verify_onnx(args.ckpt, args.onnx)
    print(f"PyTorch vs ONNX 最大误差 {max_err:.3e}，数值一致：{ok}")
    if not ok:
        raise SystemExit("数值校验未通过，请检查导出配置")


if __name__ == "__main__":
    main()
