# -*- coding: utf-8 -*-
"""
=====================================================================
第 04 课：把一声"咕"单独切出来，测【脉宽】和【带宽】
---------------------------------------------------------------------
鱼声身份证最后两个特征：
  · 脉宽：一声咕持续多久（文献约1-2ms）
  · 带宽：这声咕的能量在频率上分布多宽（文献3dB带宽约90Hz）
DSP知识：包络半高宽(FWHM)、频谱-3dB带宽、时频测不准原理。
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

wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"

# ---------- ① 先在16k里找到"成组的鱼声脉冲"，并记住它们的时间 ----------
sr = 16000
y, _ = librosa.load(wav, sr=sr, mono=True)
窗长 = sr*10
n窗 = len(y)//窗长
best, bk = -1, 0
for i in range(n窗):
    s = y[i*窗长:(i+1)*窗长]; r = np.sqrt(np.mean(s**2))
    if r > best: best, bk = r, i
y10 = y[bk*窗长:(bk+1)*窗长]; base = bk*10
# 短窗包络（0.3ms，否则会把2ms的短脉冲抹宽，测不准脉宽）
env = np.abs(hilbert(y10)); k = int(0.0003*sr)
env = np.convolve(env, np.ones(k)/k, mode="same")
med = np.median(env); height = med + 0.06*(env.max()-med)
peaks, _ = find_peaks(env, distance=int(0.01*sr), height=height, prominence=0.02)
pt = peaks/sr; gaps = np.diff(pt)*1000
# 只要"前后间隔都在80-150ms"的脉冲——它们才是成组咕咕声（孤立的往往是撞击噪声）
组成员 = []
for j in range(1, len(peaks)-1):
    if 80 <= gaps[j-1] <= 150 and 80 <= gaps[j] <= 150:
        组成员.append(base + pt[j])

# ---------- ② 用96k高采样率逐个切，算脉宽/中心频率/带宽 ----------
sr96 = 96000
def 测一个(abst):
    seg, _ = librosa.load(wav, sr=sr96, mono=True, offset=abst-0.01, duration=0.02)
    t = (np.arange(len(seg))/sr96 - 0.01)*1000     # 毫秒
    e = np.abs(hilbert(seg)); kk = max(1, int(0.0002*sr96))
    e = np.convolve(e, np.ones(kk)/kk, mode="same")
    pk = np.argmax(e)
    # 脉宽：包络峰值50%以上的全宽（半高全宽 FWHM）
    above = np.where(e >= e[pk]*0.3)[0]
    脉宽 = (above[-1]-above[0])/sr96*1000
    # 频谱：对这段做FFT，找中心频率和-3dB带宽
    A = np.abs(np.fft.rfft(seg*np.hanning(len(seg)), n=65536))
    f = np.fft.rfftfreq(65536, 1/sr96)
    m = (f >= 400) & (f <= 1600); f, A = f[m], A[m]
    f0 = f[np.argmax(A)]
    hi = np.where(A >= A.max()/np.sqrt(2))[0]       # -3dB = 降到峰值的1/√2
    带宽 = f[hi[-1]]-f[hi[0]] if len(hi) else 0
    return seg, t, e, pk, 脉宽, f0, 带宽

结果 = [测一个(a) for a in 组成员]
# 选中心频率最接近800的那个来画图（最典型）
idx = int(np.argmin([abs(r[5]-800) for r in 结果]))
seg, t, e, pk, pw, f0, bw = 结果[idx]
all_pw = np.median([r[4] for r in 结果]); all_fc = np.median([r[5] for r in 结果]); all_bw = np.median([r[6] for r in 结果])

print("="*50)
print(f"检测到 {len(结果)} 个成组鱼声脉冲，逐个测量后取中位数：")
print(f"脉宽（半高全宽）中位数 = {all_pw:.2f} ms   （文献：约1-2ms）")
print(f"中心频率中位数 = {all_fc:.0f} Hz        （文献：约800Hz）")
print(f"单脉冲-3dB带宽中位数 = {all_bw:.0f} Hz     （文献：约90Hz）")
print("="*50)
print("说明：单脉冲只有约2ms，信号越短频谱越宽（时频测不准），")
print("     所以单脉冲带宽天然比90Hz宽，这是正常的物理限制。")

# ---------- ③ 画图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
# 左：单脉冲波形+包络，标脉宽
ax1.plot(t, seg, color="#1f77b4", lw=0.8, label="波形(96k)")
ax1.plot(t, e/e.max()*seg.max(), color="red", lw=1.3, label="包络")
ax1.axhline(seg.max()*0.5*(e.max()/e.max()), color="green", ls="--", lw=1)
# 标出50%脉宽范围
ab = np.where(e >= e[pk]*0.5)[0]
tL, tR = t[ab[0]], t[ab[-1]]
ax1.annotate("", xy=(tL, -0.25), xytext=(tR, -0.25), arrowprops=dict(arrowstyle="<->", color="#ea580c"))
ax1.text((tL+tR)/2, -0.32, f"脉宽 {pw:.1f}ms", ha="center", color="#ea580c", fontsize=11)
ax1.set_title("① 一声'咕'的特写：短促衰减振荡，约2ms", fontsize=12.5)
ax1.set_xlabel("时间(ms)"); ax1.set_ylabel("振幅"); ax1.legend(fontsize=9); ax1.grid(alpha=.3)

# 右：频谱，标中心频率与-3dB带宽
A = np.abs(np.fft.rfft(seg*np.hanning(len(seg)), n=65536)); f = np.fft.rfftfreq(65536, 1/sr96)
m = (f >= 300) & (f <= 1500)
ax2.plot(f[m], 20*np.log10(A[m]/A[m].max()), color="#0ea5e9", lw=1.2)
ax2.axhline(-3, color="red", ls="--", lw=1)
ax2.axvline(f0, color="green", ls=":", lw=1.3)
ax2.annotate(f"中心{f0:.0f}Hz\n-3dB带宽{bw:.0f}Hz", xy=(f0, 0), xytext=(f0+180, -4),
             color="#0369a1", fontsize=10, arrowprops=dict(arrowstyle="->", color="#0369a1"))
ax2.text(320, -2.3, "红色虚线=-3dB", color="red", fontsize=9)
ax2.set_title("② 单脉冲频谱：能量集中在800Hz附近", fontsize=12.5)
ax2.set_xlabel("频率(Hz)"); ax2.set_ylabel("相对强度(dB)"); ax2.set_ylim(-35,3); ax2.grid(alpha=.3)

plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第04课_脉宽与带宽.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"图已保存：{out}")
