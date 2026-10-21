# -*- coding: utf-8 -*-
"""推理：加载模型，对长录音做滑窗扫描，平滑并定位成片发声时段。

支持两种后端：
- PyTorch（.pt，训练/研究环境）
- ONNX Runtime（.onnx / INT8，部署上线）
成片判定遵循"精度优先"：单窗鱼概率超阈值，再做 1 分钟居中滑动平均，
连续达到成片阈值且持续足够长才判为一个产卵时段。
"""
import os
import re
import datetime as dt

import numpy as np
import torch

from . import config as C
from .models import Cnn14
from .features import load_waveform, batch_windows_to_mel


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class FishSoundDetector:
    """统一封装 PyTorch / ONNX 两种后端。"""

    def __init__(self, model_path, backend="auto", device=None,
                 labels=None, mean=None, std=None):
        self.model_path = model_path
        self.labels = labels or C.LABELS
        self.fish_idx = self.labels.index("fish")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() and backend != "onnx" else "cpu")

        if backend == "auto":
            backend = "onnx" if model_path.lower().endswith(".onnx") else "torch"
        self.backend = backend

        if backend == "onnx":
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            if mean is not None:
                self.mean, self.std = mean, std
            else:
                # 自动读取导出时随模型落盘的标准化参数（.preprocess.json）
                from .deploy import read_preprocess_meta
                meta = read_preprocess_meta(model_path)
                if meta is not None:
                    self.mean = float(meta["mean"])
                    self.std = float(meta["std"]) + 1e-6
                    if labels is None and meta.get("labels"):
                        self.labels = list(meta["labels"])
                        self.fish_idx = self.labels.index("fish")
                else:
                    self.mean, self.std = 0.0, 1.0
                    print("[警告] 未找到 ONNX 预处理元数据，按 mean=0/std=1（不标准化）"
                          "推理；若该模型需要外部标准化，结果会错误。请改用 "
                          "03_export_onnx.py 重新导出以生成 sidecar。", flush=True)
        else:
            ck = torch.load(model_path, map_location=self.device, weights_only=False)
            self.labels = ck.get("labels", self.labels)
            self.fish_idx = self.labels.index("fish")
            self.mean = float(ck["mean"])
            self.std = float(ck["std"]) + 1e-6
            self.model = Cnn14(len(self.labels))
            self.model.load_state_dict(ck["state_dict"])
            self.model.to(self.device).eval()
            if mean is not None:
                self.mean, self.std = mean, std

    @torch.no_grad()
    def predict_proba(self, mels, batch=16):
        x = (mels - self.mean) / self.std
        if self.backend == "onnx":
            out = []
            for s in range(0, len(x), batch):
                out.append(self.session.run(
                    ["logits"], {"mel": x[s:s + batch].astype(np.float32)})[0])
            return softmax(np.concatenate(out))
        xt = torch.from_numpy(x).float().to(self.device)
        out = []
        for s in range(0, len(xt), batch):
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits, _ = self.model(xt[s:s + batch])
            out.append(torch.softmax(logits, 1).float().cpu())
        return torch.cat(out).numpy()

    def scan_waveform(self, y, win_sec=C.WIN_SEC, step_sec=C.SEG_SEC, batch=16):
        """返回 (中点秒, 每窗鱼概率, 每窗四类概率)。"""
        mels, starts = batch_windows_to_mel(y, win_sec, step_sec)
        proba = self.predict_proba(mels, batch)
        mid_sec = starts / C.SAMPLE_RATE + win_sec / 2
        return mid_sec, proba[:, self.fish_idx], proba

    def scan_file(self, path, batch=16):
        y = load_waveform(path, clip_sec=None)  # 长录音不裁剪
        return self.scan_waveform(y, batch=batch)


def smooth(x, win=C.SMOOTH_WIN):
    """居中滑动平均。"""
    if len(x) < win:
        return np.full_like(x, x.mean())
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def detect_segments(mid_sec, fish_proba, fish_thresh=C.FISH_THRESH,
                    seg_thresh=C.SEG_THRESH, smooth_win=C.SMOOTH_WIN,
                    min_minutes=1.5, gap_windows=20):
    """从逐窗鱼概率提取成片产卵时段。返回 [(开始分钟, 结束分钟, 峰值), ...]。"""
    sm = smooth(fish_proba, smooth_win)
    mask = sm >= seg_thresh
    segments = []
    i, n = 0, len(mask)
    step_min = (C.SEG_SEC / 60.0)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j < n and gap <= gap_windows:
            if mask[j]:
                gap = 0
            else:
                gap += 1
            j += 1
        j = max(i + 1, j - gap)
        duration_min = (j - i) * step_min
        if duration_min >= min_minutes:
            seg = sm[i:j]
            peak_i = i + int(np.argmax(seg))
            segments.append((mid_sec[i] / 60, mid_sec[min(j, n) - 1] / 60,
                             float(fish_proba[i:j].max()), float(seg[peak_i])))
        i = j
    return segments


def parse_recording_timestamp(filename):
    """解析文件名里的 @YYYYMMDD_HHMMSS@MID 时间戳。"""
    m = re.search(r"@(\d{8})_(\d{6})@MID", filename)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
