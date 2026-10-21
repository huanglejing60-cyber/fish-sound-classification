# -*- coding: utf-8 -*-
"""
=====================================================================
第 05 课：自动鱼声检测器——用"四道关"从噪声里捞出鱼声
---------------------------------------------------------------------
前4课：我们知道哪段有鱼声，手动量它的特征。
这节课：给电脑一整段30分钟录音，让它【自动】找出哪几分钟有鱼声。

思路：先用一个【宽松】的能量门槛，把所有"响的东西"都当成候选
（宁多勿漏）；再用第1-4课量的特征，设四道关卡逐一筛选：
  关1 频率关：脉冲主频率必须在 650~1000Hz（鱼声约800Hz）
  关2 窄带关：能量要集中在峰频附近（撞击声是"什么频率都有"的宽带）
  关3 成组关：脉冲要间隔80~150ms成串出现（撞击是孤零零的"咚"一声）
最后：成组出现的地方，才判定为鱼声。
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
sr = 16000
y, _ = librosa.load(wav, sr=sr, mono=True)

块秒 = 10
块长 = sr*块秒
n块 = len(y)//块长

def 检测一块(s):
    """对一个10秒的块，返回：候选数、过前两关数、成组对数、是否判鱼"""
    # ---- 第0步：包络 + 宽松门槛找候选（宁多勿漏）----
    env = np.abs(hilbert(s)); k = int(0.005*sr)
    env = np.convolve(env, np.ones(k)/k, mode="same")
    med = np.median(env); h = med + 0.08*(env.max()-med)
    pk, _ = find_peaks(env, distance=int(0.02*sr), height=h, prominence=0.02)
    if len(pk) == 0:
        return 0, 0, 0, 0
    过关 = []
    for p in pk:
        a = max(0, p-int(0.008*sr)); b = p+int(0.008*sr)
        seg = s[a:b]*np.hanning(b-a)              # 切出这个脉冲
        nfft = 2048
        A = np.abs(np.fft.rfft(seg, nfft)); f = np.fft.rfftfreq(nfft, 1/sr)
        m = (f >= 300) & (f <= 1500); f, A = f[m], A[m]
        f0 = f[np.argmax(A)]                       # 峰频
        # 关1 频率关：峰频在650~1000Hz（排除低频水流、高频撞击）
        if not (650 <= f0 <= 1000):
            continue
        # 关2 窄带关：峰频±120Hz内的能量占比要够高（排除"什么频率都有"的宽带撞击）
        窄带占比 = np.sum(A[(f >= f0-120) & (f <= f0+120)])/np.sum(A)
        if 窄带占比 > 0.5:
            过关.append(p)
    # 关3 成组关：过关脉冲里，间隔80~150ms的相邻对有几个
    组数 = 0
    if len(过关) >= 2:
        d = np.diff(np.array(过关)/sr)*1000
        组数 = int(np.sum((d >= 80) & (d <= 150)))
    判鱼 = 1 if 组数 >= 2 else 0
    return len(pk), len(过关), 组数, 判鱼

# 逐块检测
候选数, 过关数, 组数, 判鱼 = [], [], [], []
for bi in range(n块):
    s = y[bi*块长:(bi+1)*块长]
    c, g, z, j = 检测一块(s)
    候选数.append(c); 过关数.append(g); 组数.append(z); 判鱼.append(j)
分钟 = np.arange(n块)*块秒/60
候选数, 组数, 判鱼 = map(np.array, (候选数, 组数, 判鱼))

# 漏斗统计
print("="*52)
print(f"30分钟一共：")
print(f"  第0步 宽松候选（所有响声）：{候选数.sum()} 个")
print(f"  关1+关2 频率&窄带过滤后：{int(np.sum(过关数))} 个")
print(f"  关3 成组确认的鱼声块：{判鱼.sum()} / {n块} 块（每块10秒）")
鱼分钟 = 分钟[判鱼 == 1]
if len(鱼分钟):
    print(f"  判定有鱼声的时间段：第 {鱼分钟.min():.0f}~{鱼分钟.max()+0.17:.0f} 分钟之间")
print("="*52)
print("（录音18:51开始，第23分钟起≈19:14，即傍晚产卵大合唱）")

# 画图：上=候选很多，下=四道关后只剩高峰
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.bar(分钟, 候选数, width=0.15, color="#94a3b8")
ax1.set_title("第①步：宽松门槛——所有'响声'都算候选（水流/撞击/鱼声全混在一起，看不出规律）", fontsize=12)
ax1.set_ylabel("候选脉冲数")
ax2.bar(分钟, 组数, width=0.15, color="#dc2626")
for x in 分钟[判鱼 == 1]:
    ax2.axvspan(x, x+块秒/60, color="#fecaca", alpha=0.5)
ax2.set_title("第②步：四道关后——只有傍晚产卵高峰留下，前面的噪声全被挡掉", fontsize=12)
ax2.set_ylabel("成组鱼声对数"); ax2.set_xlabel("录音开始后的时间（分钟）")
ax2.annotate("产卵大合唱\n19:14起", xy=(23, ax2.get_ylim()[1]*0.8), fontsize=12, color="#b91c1c",
             arrowprops=dict(arrowstyle="->", color="#b91c1c"))
plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第05课_自动检测_30分钟产卵高峰.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("图已保存：", out)
