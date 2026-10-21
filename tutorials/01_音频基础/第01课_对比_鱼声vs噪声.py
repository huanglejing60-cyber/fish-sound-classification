# -*- coding: utf-8 -*-
"""
=====================================================================
第 01 课（下）：一张图对比【成鱼鱼声】和【鱼苗段噪声】
---------------------------------------------------------------------
让眼睛建立直觉：同样是"最响的10秒"，
鱼声和噪声在波形、语谱上长得到底哪里不一样。
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

sr0 = 16000
窗秒 = 10

# 两个录音：一个是会叫的成鱼（傍晚产卵段），一个是3.12蝌蚪状鱼苗
音频 = {
    "成鱼（有鱼声·傍晚产卵）": r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav",
    "鱼苗3.12（蝌蚪状·应无鱼声）": r"E:\大黄鱼数据\#30053#[026]@20250312_154642\#30053#[026]@20250312_154642@MID@96Ksps.wav",
}

def 取最响10秒(路径):
    y, sr = librosa.load(路径, sr=sr0, mono=True)
    窗长 = sr * 窗秒
    n窗 = len(y) // 窗长
    最响, 最响窗 = -1, 0
    for i in range(n窗):
        段 = y[i*窗长:(i+1)*窗长]
        rms = np.sqrt(np.mean(段**2))
        if rms > 最响:
            最响, 最响窗 = rms, i
    return y[最响窗*窗长:(最响窗+1)*窗长], sr, 最响窗*窗秒

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col, (名称, 路径) in enumerate(音频.items()):
    print(f"正在分析：{名称} ……")
    y, sr, 起点 = 取最响10秒(路径)
    t = np.arange(len(y)) / sr
    主色 = "#d6336c" if col == 0 else "#1f77b4"

    # 上：波形
    axw = axes[0, col]
    axw.plot(t, y, color=主色, lw=0.4)
    axw.set_title(f"{名称}\n波形（整段第{起点//60:.0f}分{起点%60:.0f}秒起）", fontsize=12)
    axw.set_xlabel("时间(秒)"); axw.set_ylabel("振幅")
    axw.set_xlim(0, 窗秒); axw.set_ylim(-1.05, 1.05); axw.grid(alpha=0.3)

    # 下：语谱
    axs = axes[1, col]
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
                                   ax=axs, cmap="magma", vmin=-60, vmax=0)
    axs.set_ylim(0, 3000)
    axs.axhline(800, color="cyan", ls="--", lw=1.2)
    axs.set_title("语谱（青色虚线=800Hz）", fontsize=12)

plt.tight_layout()
out = r"E:\大黄鱼分析\学习\第01课_对比_鱼声vs噪声.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\n对比图已保存：{out}")
print("对比要点：左边800Hz附近是【一簇簇规律的垂直亮纹】；右边没有这种规律亮纹（安静/水流弥漫/低频隆隆）。")
