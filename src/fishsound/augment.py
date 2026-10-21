# -*- coding: utf-8 -*-
"""数据增强（训练时随机生效，验证集绝不增强）。

- 波形级：随机增益、时间平移、以一定概率叠加真实背景噪声
- 频谱级 SpecAugment：随机遮挡若干频带 / 时间帧
作用是"故意不让模型死记训练样本"，压低训练分、换陌生数据上的鲁棒性，
是防过拟合的保险；它不凭空造信息，小样本上提分有限（数据才是上限）。
"""
import numpy as np

from . import config as C


# ---------------- 波形级增强 ----------------
def random_gain(y, low=0.6, high=1.2, rng=None):
    rng = rng or np.random
    return y * rng.uniform(low, high)


def random_time_shift(y, max_sec=0.4, sr=C.SAMPLE_RATE, rng=None):
    rng = rng or np.random
    shift = int(rng.uniform(-max_sec, max_sec) * sr)
    return np.roll(y, shift)


def maybe_mix_noise(y, noise_pool, prob=0.5, low=0.1, high=0.4, rng=None):
    """以 prob 概率从噪声池随机取一条背景声按小系数叠加。"""
    rng = rng or np.random
    if rng.rand() > prob or len(noise_pool) == 0:
        return y
    n = noise_pool[rng.randint(len(noise_pool))]
    n = np.resize(n, len(y))
    return y + rng.uniform(low, high) * n


# ---------------- 频谱级 SpecAugment ----------------
def _mask_axis(mel, axis, n_masks, max_width, rng):
    out = mel.copy()
    size = mel.shape[axis]
    for _ in range(n_masks):
        if max_width <= 0:
            continue
        w = rng.randint(0, max_width + 1)
        if w == 0:
            continue
        start = rng.randint(0, max(size - w, 1))
        if axis == 0:
            out[start:start + w, :] = mel.min()
        else:
            out[:, start:start + w] = mel.min()
    return out


def spec_augment(mel, n_freq=1, freq_width=12, n_time=1, time_width=50, rng=None):
    """在 mel 上随机遮频带（轴0）和时间帧（轴1）。"""
    rng = rng or np.random
    n_freq = rng.randint(1, n_freq + 2)
    n_time = rng.randint(1, n_time + 2)
    out = _mask_axis(mel, 0, n_freq, freq_width, rng)
    out = _mask_axis(out, 1, n_time, time_width, rng)
    return out


def make_mel_augmenter(seed=None):
    """返回 mel->mel 的增强函数，供训练 Dataset 使用。"""
    rng = np.random.RandomState(seed)

    def augment(mel):
        return spec_augment(mel, rng=rng)

    return augment
