# -*- coding: utf-8 -*-
"""部署：PyTorch 导出 ONNX、数值校验、静态 INT8 量化（混合精度）、延迟基准。

完整部署链路：
  .pt -> torch.onnx.export -> .onnx(fp32) -> quantize_static -> INT8(混合精度)
关键经验：朴素量化会掉精度，把第一层卷积和分类头保留 fp32 后精度无损。
"""
import os
import json
import time
import shutil
import tempfile

import numpy as np
import torch

from . import config as C
from .models import Cnn14, InferenceWrapper


def meta_path_for(onnx_path):
    """与 ONNX 配套的预处理元数据文件路径。"""
    return onnx_path + ".preprocess.json"


def write_preprocess_meta(onnx_path, ck):
    """导出 ONNX 时把标准化参数/标签写成 sidecar，供部署端读取。

    ONNX 图本身不含标准化（输入应为 (mel-mean)/std 后的 log-mel），
    因此把 mean/std 随模型一起落盘，避免跨语言部署时漏掉预处理。
    """
    meta = {
        "mean": float(ck["mean"]),
        "std": float(ck["std"]),
        "labels": list(ck.get("labels", C.LABELS)),
        "input": "standardized log-mel: (mel - mean)/std",
    }
    with open(meta_path_for(onnx_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path_for(onnx_path)


def read_preprocess_meta(onnx_path):
    p = meta_path_for(onnx_path)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_torch_for_export(ckpt_path, device="cpu", model_name=None):
    """加载 checkpoint 并套上部署外壳。

    model_name: 'cnn14'（默认）或轻量模型 'dscnn' / 'mobilenet'。
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels = ck.get("labels", C.LABELS)
    if model_name is None or model_name == "cnn14":
        model = Cnn14(len(labels))
    else:
        from .models import build_lightweight
        model = build_lightweight(model_name, classes=len(labels))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return InferenceWrapper(model).eval(), ck


def export_onnx(ckpt_path, onnx_path, opset=17, model_name=None):
    """导出 ONNX（传统 TorchScript 导出器，dynamo=False 更稳）。"""
    wrapper, ck = load_torch_for_export(ckpt_path, model_name=model_name)
    dummy = torch.randn(1, C.N_MELS, C.N_FRAMES, dtype=torch.float32)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["mel"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    write_preprocess_meta(onnx_path, ck)
    return onnx_path


def verify_onnx(ckpt_path, onnx_path, atol=1e-4, model_name=None):
    """对比 PyTorch 与 ONNX 对同一输入的输出，返回最大误差。"""
    import onnxruntime as ort
    wrapper, _ = load_torch_for_export(ckpt_path, model_name=model_name)
    x = torch.randn(1, C.N_MELS, C.N_FRAMES)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"mel": x.numpy().astype(np.float32)})[0]
    max_err = float(np.abs(ref - out).max())
    return max_err, bool(np.allclose(ref, out, atol=atol, rtol=1e-3))


class _CalibrationReader:
    def __init__(self, x):
        self.it = iter([{"mel": x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        try:
            return next(self.it)
        except StopIteration:
            return None


def quantize_mixed_int8(onnx_path, calib_mels, out_path,
                        sensitive_nodes=None, method="entropy"):
    """静态量化（QDQ），敏感层保留 fp32（混合精度）。

    量化工具在中文路径下会找不到临时文件，因此放到纯英文临时目录做完再复制。
    """
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
        CalibrationMethod)
    sensitive_nodes = (sensitive_nodes if sensitive_nodes is not None
                       else C.QUANT_SENSITIVE_NODES)
    method = (CalibrationMethod.Entropy if method == "entropy"
              else CalibrationMethod.MinMax)

    class Cal(CalibrationDataReader):
        def __init__(self, x):
            if hasattr(x, "detach"):  # 兼容 torch tensor
                x = x.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            self.en = iter([{"mel": x[i:i + 1]} for i in range(len(x))])

        def get_next(self):
            try:
                return next(self.en)
            except StopIteration:
                return None

    # 校验敏感层节点真实存在；节点名前缀（/model vs /m）取决于导出包装层，
    # 这里按后缀自动重映射，找不到唯一匹配就直接报错——避免静默退化成全量化。
    import onnx
    existing = {n.name for n in onnx.load(onnx_path).graph.node}
    resolved = []
    for name in sensitive_nodes:
        if name in existing:
            resolved.append(name)
            continue
        suffix = name.split("/", 2)[-1]  # 如 conv_block1/conv1/Conv
        cand = sorted(e for e in existing if e.endswith(suffix))
        if len(cand) == 1:
            print(f"[量化] 排除节点 {name} -> 自动匹配 {cand[0]}", flush=True)
            resolved.append(cand[0])
        else:
            raise ValueError(
                f"敏感层节点 {name} 在 ONNX 图中找不到唯一匹配，候选={cand}；"
                f"请检查 QUANT_SENSITIVE_NODES，否则混合精度会失效。")

    tmp = tempfile.mkdtemp(prefix="onnxq_")
    src = os.path.join(tmp, "model.onnx")
    dst = os.path.join(tmp, "quant.onnx")
    shutil.copy(onnx_path, src)
    quantize_static(
        src, dst, Cal(calib_mels), quant_format=QuantFormat.QDQ,
        per_channel=True, weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8, calibrate_method=method,
        nodes_to_exclude=resolved)
    shutil.copy(dst, out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def benchmark(path, backend=None, batch=1, n=80, warmup=8, model_name=None):
    """测单批推理延迟，返回 (中位数ms, p95ms)。"""
    x = np.random.randn(batch, C.N_MELS, C.N_FRAMES).astype(np.float32)
    if path.lower().endswith(".onnx") or backend == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        def run():
            sess.run(["logits"], {"mel": x})
    else:
        wrapper, _ = load_torch_for_export(path, model_name=model_name)

        def run():
            with torch.no_grad():
                wrapper(torch.from_numpy(x))
    for _ in range(warmup):
        run()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        run()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))
