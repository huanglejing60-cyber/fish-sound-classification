# -*- coding: utf-8 -*-
"""
=====================================================================
第 02 课：把鱼声"放大"，亲手测量它的【频率】
---------------------------------------------------------------------
目标：在语谱上已经看到约800Hz的亮纹，这节课用频谱（FFT）把它算出来，
     验证"大黄鱼发声谱峰约800Hz"。
DSP理论：把一段信号做FFT，就得到"各频率的强度"；最强的那个频率就是谱峰。
=====================================================================
"""
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 成鱼傍晚产卵段
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr = 16000
y, sr = librosa.load(wav, sr=sr, mono=True)

# ① 自动找最响的10秒（和第1课一样）
窗长 = sr * 10
n窗 = len(y) // 窗长
最响, 最响窗 = -1, 0
for i in range(n窗):
    seg = y[i*窗长:(i+1)*窗长]
    r = np.sqrt(np.mean(seg**2))
    if r > 最响:
        最响, 最响窗 = r, i
y10 = y[最响窗*窗长:(最响窗+1)*窗长]

# ② 在这10秒里，只保留"鱼叫的瞬间"：
#    能量超过阈值的采样点（即波形上扎高的尖刺），其他时间设为0。
#    这样算频谱时，水流背景的影响被压掉，留下的主要是鱼声。
能量 = y10**2
阈值 = np.percentile(能量, 99)          # 取最响的3%采样点（鱼声尖刺）
y鱼 = np.where(能量 >= 阈值, y10, 0.0)

# ③ 对"鱼叫瞬间"做 FFT，得到频谱
#    加一个汉宁窗（np.hanning），减少频谱泄漏，让谱峰更准
N = len(y鱼)
窗 = np.hanning(N)
频谱 = np.fft.rfft(y鱼 * 窗)
幅度 = np.abs(频谱) / N
频率 = np.fft.rfftfreq(N, d=1/sr)

# 只看 0–2000Hz（鱼声在这里）
mask = (频率 >= 100) & (频率 <= 2000)
f0, a0 = 频率[mask], 幅度[mask]
谱峰频率 = f0[np.argmax(a0)]
print(f"===== 测量结果 =====")
print(f"这段鱼声的谱峰频率 ≈ {谱峰频率:.0f} Hz   （文献值：约800Hz）")

# ④ 画图：左=保留下来的鱼声尖刺波形，右=频谱（标出谱峰）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
t = np.arange(len(y10))/sr
ax1.plot(t, y鱼, color="#d6336c", lw=0.5)
ax1.set_title("只保留鱼叫瞬间（最响3%采样点）")
ax1.set_xlabel("时间(秒)"); ax1.set_ylabel("振幅"); ax1.set_xlim(0,10); ax1.grid(alpha=0.3)

ax2.plot(f0, a0, color="#0ea5e9", lw=1)
ax2.axvline(谱峰频率, color="red", ls="--", lw=1.2)
ax2.axvline(800, color="green", ls=":", lw=1.5)
ax2.annotate(f"实测谱峰 {谱峰频率:.0f}Hz", xy=(谱峰频率, a0.max()),
             xytext=(谱峰频率+150, a0.max()*0.9), color="red", fontsize=11,
             arrowprops=dict(arrowstyle="->", color="red"))
ax2.text(800, a0.max()*0.75, "文献800Hz(绿点线)", color="green", fontsize=10, ha="center")
ax2.set_title("频谱图：鱼声能量主要集中在哪个频率")
ax2.set_xlabel("频率(Hz)"); ax2.set_ylabel("强度")
ax2.set_xlim(0, 2000); ax2.grid(alpha=0.3)

plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第02课_鱼声谱峰测量.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"图已保存：{out}")
