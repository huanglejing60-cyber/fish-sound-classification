# -*- coding: utf-8 -*-
"""音频读取与 log-mel 特征提取（纯 numpy/scipy 实现）。

数据流：wav -> soundfile 解码 -> soxr 重采样到 32k -> 取/补 3 秒
        -> 补零中心 STFT -> 功率谱 -> Slaney mel 滤波器 -> log(dB)。

为什么不用 librosa：在 librosa 1.0 / numba 0.67 的部分环境里，
``librosa.load`` 的重采样和 ``librosa.filters.mel`` / ``melspectrogram``
会在多线程 JIT 初始化时长时间卡死（3 秒音频数十秒不返回）。这里改用
soundfile + soxr + numpy 手写，参数与 librosa 0.10/1.0 默认口径严格对齐
（periodic hann 窗、center 补零、htk=False、Slaney 归一化、功率谱 dB），
结果与训练 CNN14 时一致，而且更快、更易移植（便于后续 C++ 部署）。
"""
from functools import lru_cache

import numpy as np
import soundfile as sf
import soxr
from scipy.signal.windows import hann

from . import config as C


# ---------------- 读取 / 重采样 ----------------

def load_waveform(path, sr=C.SAMPLE_RATE, clip_sec=C.WIN_SEC):
    """读单声道音频并（可选）重采样、裁剪/补齐到固定长度。"""
    y, orig_sr = sf.read(path, dtype="float32")
    if y.ndim > 1:  # 多声道 -> 单声道（取平均）
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = soxr.resample(y, orig_sr, sr, quality="HQ")
    y = np.asarray(y, dtype=np.float32)
    if clip_sec is not None:
        n = int(sr * clip_sec)
        y = np.pad(y, (0, max(0, n - len(y))))[:n]
    return y


# ---------------- Slaney mel 滤波器（htk=False, norm='slaney'） ----------------

def _hz_to_mel(f):
    f_sp = 200.0 / 3.0
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    f = np.asarray(f, dtype=np.float64)
    m = f / f_sp
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep,
                    m)


def _mel_to_hz(m):
    f_sp = 200.0 / 3.0
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    m = np.asarray(m, dtype=np.float64)
    h = f_sp * m
    return np.where(m >= min_log_mel,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)),
                    h)


@lru_cache(maxsize=None)
def _mel_basis(sr, n_fft, n_mels, fmin, fmax):
    """返回 (n_mels, 1+n_fft//2) 的 Slaney 三角形滤波器矩阵（缓存）。"""
    n_freq = 1 + n_fft // 2
    weights = np.zeros((n_mels, n_freq), dtype=np.float64)
    fftfreqs = np.linspace(0, sr / 2.0, n_freq)
    m_min, m_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    pts = _mel_to_hz(m_min + np.arange(n_mels + 2) * (m_max - m_min) / (n_mels + 1))
    fdiff = np.diff(pts)
    rslopes = pts[:, None] - fftfreqs[None, :]
    for i in range(n_mels):
        lower = -rslopes[i] / fdiff[i]
        upper = rslopes[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (pts[2:n_mels + 2] - pts[:n_mels])  # Slaney 归一化
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def _stft(y, n_fft, hop):
    """center=True、两端补零的 STFT，返回 (1+n_fft//2, n_frames) 复谱。"""
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad), mode="constant")
    n_frames = 1 + (len(yp) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        yp, shape=(n_fft, n_frames),
        strides=(yp.strides[0], hop * yp.strides[0]))
    window = hann(n_fft, sym=False).astype(np.float32)  # periodic hann
    return np.fft.rfft(frames * window[:, None], n_fft, axis=0)


def waveform_to_mel(y, sr=C.SAMPLE_RATE):
    """一段波形 -> 固定帧数 (N_MELS, N_FRAMES) 的 log-mel（功率谱 dB）。"""
    y = np.asarray(y, dtype=np.float32)
    spec = _stft(y, C.N_FFT, C.HOP_LENGTH)
    power = spec.real ** 2 + spec.imag ** 2
    mel = _mel_basis(sr, C.N_FFT, C.N_MELS, C.FMIN, C.FMAX) @ power
    S = 10.0 * np.log10(np.maximum(mel, 1e-10)).astype(np.float32)
    if S.shape[1] >= C.N_FRAMES:
        S = S[:, :C.N_FRAMES]
    else:
        S = np.pad(S, ((0, 0), (0, C.N_FRAMES - S.shape[1])))
    return S


def load_mel(path):
    """直接从文件路径读 3 秒 log-mel。"""
    return waveform_to_mel(load_waveform(path))


def batch_windows_to_mel(y, win_sec=C.WIN_SEC, step_sec=C.SEG_SEC, sr=C.SAMPLE_RATE):
    """长录音滑窗切分 -> 批量 mel，返回 (mels, 起始样本点)。"""
    win = int(win_sec * sr)
    step = int(step_sec * sr)
    starts = list(range(0, len(y) - win + 1, step))
    mels = np.stack([
        waveform_to_mel(y[s:s + win], sr=sr) for s in starts
    ]).astype(np.float32)
    return mels, np.array(starts)
