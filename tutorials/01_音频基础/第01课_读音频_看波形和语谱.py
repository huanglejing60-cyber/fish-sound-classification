# -*- coding: utf-8 -*-
"""
=====================================================================
第 01 课：读入一段鱼声录音，画出【波形图】和【语谱图】
---------------------------------------------------------------------
这是你分析一切音频的起点。先不用懂每个函数，跟着注释跑通、看图，
后面几课我会逐个拆开讲。
需要的库：numpy(算数据)、librosa(读音频/做频谱)、matplotlib(画图)、soundfile(读wav)
=====================================================================
"""

# ---------- 第①步：导入工具包 ----------
# 可以理解为：把别人造好的"工具箱"搬进来用
import numpy as np                      # 专门处理数字数组（音频本质就是一串数字）
import librosa                        # 音频分析最常用的库（读音频、做频谱）
import librosa.display                # 画语谱图要用到它的显示函数
import soundfile as sf                # 读 wav 文件（也能查采样率）
import matplotlib
matplotlib.use("Agg")                 # 让图直接保存成图片文件，不依赖弹窗
import matplotlib.pyplot as plt       # 画图库，像 MATLAB 里的 plot

# 让图里的中文不乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 第②步：指定要读的录音文件（3.13傍晚18:51产卵高峰段）----------
wav_path = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"

# 原文件是 96000Hz(每秒9.6万个点)，鱼声频率在800Hz左右，
# 我们只需要分析到几千Hz，所以降到16000Hz足够，数据量小、跑得快。
目标采样率 = 16000
看多少秒 = 5

# ---------- 第③步：先自动找到"鱼声最密集的10秒" ----------
print("正在读取整段录音，请稍等……")
# 读入整段（30分钟），并降采样到16000
y_full, sr = librosa.load(wav_path, sr=目标采样率, mono=True)
print(f"读完了：采样率={sr}Hz，整段时长={len(y_full)/sr/60:.1f}分钟，共{len(y_full)}个采样点")

# 把整段切成一个个10秒窗口，算每个窗口的"平均响度"，找最响的那段
# （产卵期鱼声合唱，最响的窗口基本就是鱼声最密集的地方）
窗长 = sr * 看多少秒
n窗 = len(y_full) // 窗长
最响 = -1; 最响窗 = 0
for i in range(n窗):
    段 = y_full[i*窗长:(i+1)*窗长]
    rms = np.sqrt(np.mean(段**2))      # RMS=均方根，代表这一段的平均响度
    if rms > 最响:
        最响 = rms; 最响窗 = i
起点秒 = 最响窗 * 看多少秒
y = y_full[最响窗*窗长:(最响窗+1)*窗长]   # 取出这10秒
print(f"鱼声最密集的10秒：从整段第 {起点秒//60:.0f}分{起点秒%60:.0f}秒 开始")

# ---------- 第④步：画波形图（上）和语谱图（下）----------
时间轴 = np.arange(len(y)) / sr          # 每个采样点对应的时间(秒)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

# 波形图：横轴=时间，纵轴=声音的振幅(上下振动幅度)。
# 鱼叫时能看到一簇一簇"上下扎得很高"的刺；安静时贴着中间一条线。
ax1.plot(时间轴, y, color="#1f77b4", lw=0.5)
ax1.set_title("① 波形图：一簇一簇的尖刺 = 鱼在叫（咕咕声）", fontsize=13)
ax1.set_xlabel("时间 (秒)"); ax1.set_ylabel("振幅")
ax1.set_xlim(0, 看多少秒); ax1.grid(alpha=0.3)

# 语谱图：横轴=时间，纵轴=频率(音高)，颜色越亮=那个频率越响。
# 鱼声 = 一条条垂直的亮纹，且集中在约800Hz附近。
D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))   # STFT=短时傅里叶变换，把信号变到频率维度
DB = librosa.amplitude_to_db(D, ref=np.max)               # 转成分贝(dB)，颜色更好看
img = librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis="time",
                               y_axis="hz", ax=ax2, cmap="magma", vmin=-60, vmax=0)
ax2.set_ylim(0, 3000)        # 只看0-3000Hz，鱼声800Hz就在这里
ax2.axhline(800, color="cyan", ls="--", lw=1.2)            # 画一条800Hz参考线
ax2.text(0.1, 860, "鱼声约800Hz", color="cyan", fontsize=10)
ax2.set_title("② 语谱图：一条条垂直亮纹、集中在800Hz附近 = 鱼声", fontsize=13)
fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="响度")

plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第01课_鱼声波形与语谱.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\n图已保存到：{out}")
print("请打开这张图，对照上面的文字看一看～")
