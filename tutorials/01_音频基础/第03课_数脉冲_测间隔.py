# -*- coding: utf-8 -*-
"""
=====================================================================
第 03 课：数鱼"咕"了几下，测【脉冲间隔】
---------------------------------------------------------------------
目标：产卵声是"咕-咕-咕"的双/三脉冲，文献说相邻脉冲间隔约100-130ms。
     这节课用"包络检波 + 峰值检测"把每一下找出来，亲手验证这个间隔。
DSP知识：包络检波（你通信原理里AM解调的思路）+ 峰值检测。
=====================================================================
"""
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- ① 读成鱼产卵段，取最响10秒 ----------
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr = 16000
y, _ = librosa.load(wav, sr=sr, mono=True)
窗长 = sr * 10
n窗 = len(y) // 窗长
best, bk = -1, 0
for i in range(n窗):
    seg = y[i*窗长:(i+1)*窗长]
    r = np.sqrt(np.mean(seg**2))
    if r > best:
        best, bk = r, i
y10 = y[bk*窗长:(bk+1)*窗长]
t = np.arange(len(y10)) / sr

# ---------- ② 算"包络" ----------
# 原始波形上下快速振荡，直接找峰会在一个"咕"里数出很多个。
# 包络 = 把振荡包成一条表示声音强弱的平滑曲线（hilbert取幅度，再5ms滑动平均）。
analytic = hilbert(y10)                 # 解析信号，它的模就是瞬时幅度
env = np.abs(analytic)
k = int(0.005 * sr)
env = np.convolve(env, np.ones(k)/k, mode="same")

# ---------- ③ 在包络上找每一个"咕"（峰值）----------
# distance: 两个峰至少隔20ms（防止一个咕被数成两个）
# height:   峰要高于底噪（中位数+一点余量），过滤水流
# prominence: 峰要明显比周围"鼓"出来
med = np.median(env)
height = med + 0.06 * (env.max() - med)
peaks, _ = find_peaks(env, distance=int(0.02*sr), height=height, prominence=0.02)
peak_t = peaks / sr                     # 每个脉冲的时刻(秒)
gaps_ms = np.diff(peak_t) * 1000        # 相邻脉冲间隔(毫秒)

# ---------- ④ 打印结果，和文献对照 ----------
print("=" * 46)
print(f"这10秒里共检测到 {len(peaks)} 下鱼声脉冲")
print(f"相邻脉冲间隔中位数 = {np.median(gaps_ms):.0f} ms   （文献产卵声：约100-130ms）")
组内 = ((gaps_ms >= 80) & (gaps_ms <= 150)).sum()   # 同一次发声里的双/三脉冲
停顿 = (gaps_ms > 300).sum()                        # 两次发声之间的停顿
print(f"其中间隔80-150ms的（咕咕紧挨着=双/三脉冲）：{组内} 个")
print(f"间隔>300ms的（两次发声之间的停顿）：{停顿} 个")
print("=" * 46)

# ---------- ⑤ 画图：上=10秒全景，下=放大看清脉冲和间隔 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

# 上：全景，红点=每个脉冲
ax1.plot(t, y10, color="#94a3b8", lw=0.4, label="原始波形")
ax1.plot(t, env, color="#d6336c", lw=1.2, label="包络(声音强弱轮廓)")
ax1.plot(peak_t, env[peaks], "v", color="red", ms=7, label="检测到的每一下")
ax1.set_title(f"① 10秒产卵合唱：共{len(peaks)}下脉冲（红点），间隔中位数{np.median(gaps_ms):.0f}ms", fontsize=13)
ax1.set_xlabel("时间(秒)"); ax1.set_ylabel("振幅")
ax1.set_xlim(0, 10); ax1.legend(loc="upper right", fontsize=9); ax1.grid(alpha=0.3)

# 下：放大 3.7-4.6 秒，看清一组"咕-咕-咕"
x0, x1 = 3.7, 4.6
m = (t >= x0) & (t <= x1)
ax2.plot(t[m], y10[m], color="#1f77b4", lw=0.7)
sel = peak_t[(peak_t >= x0) & (peak_t <= x1)]
for pt0 in sel:
    ax2.axvline(pt0, color="red", ls="--", lw=1, alpha=0.7)
# 标注相邻间隔
for a in range(len(sel)-1):
    p1, p2 = sel[a], sel[a+1]
    gap = (p2 - p1) * 1000
    ax2.annotate("", xy=(p1, 0.9), xytext=(p2, 0.9),
                 arrowprops=dict(arrowstyle="<->", color="#ea580c"))
    ax2.text((p1+p2)/2, 0.96, f"{gap:.0f}ms", ha="center", color="#ea580c", fontsize=10)
ax2.set_title("② 放大看：每次发声是紧挨着的'咕-咕-咕'，间隔约90-130ms（产卵双/三脉冲）", fontsize=13)
ax2.set_xlabel("时间(秒)"); ax2.set_ylabel("振幅")
ax2.set_xlim(x0, x1); ax2.set_ylim(-1.1, 1.1); ax2.grid(alpha=0.3)

plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第03课_脉冲间隔测量.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"图已保存：{out}")
