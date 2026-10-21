# -*- coding: utf-8 -*-
"""
=====================================================================
第 18 课 a：数据增强长什么样？（只画图、不训练）
---------------------------------------------------------------------
你的训练集真鱼只有32个，模型"见得少"。数据增强 = 拿一段真实声音，
人工变出很多个"略有不同但还是同一种声音"的样本，让模型见多识广、
不死记硬背（防过拟合）。这是音频深度学习的标准操作、面试高频。

本脚本拿一段真鱼，展示6种常用增强在语谱图上的变化：
  波形级（对原始声音做）：加背景噪声、时间平移、音量缩放
  频谱级（对mel图做）：SpecAugment频率遮挡、时间遮挡
  混合：Mixup（鱼+别的声音，标签也按比例混合）
=====================================================================
"""
import os, csv
import numpy as np, librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
ROOT=r"E:\大黄鱼分析\dataset_train_v1"
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300; DUR=3
rng=np.random.default_rng(42)

def 读波形(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    y=np.pad(y,(0,max(0,SR*DUR-len(y))))[:SR*DUR]; return y
def 转mel(y):
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

# 每类取一个样本（fish选真实产卵段，water/stress各取一个）
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
def 取一个(类别, 优先含=None):
    cand=[r for r in rows if r["类别"]==类别]
    r=next((r for r in cand if 优先含 and 优先含 in r["文件名"]), cand[0])
    return r,读波形(os.path.join(ROOT,类别,r["文件名"]))
rf,yf=取一个("fish","205627"); rw,yw=取一个("water"); rs,ys=取一个("stress")
print("真鱼样本:",rf["文件名"]); print("水流样本:",rw["文件名"]); print("应激样本:",rs["文件名"])

# ---------- 波形级增强 ----------
def 加噪(y,noise,k=0.3):  # 叠真实背景噪声（对水下场景特别有用）
    n=noise[:len(y)]*k
    return (y+n)/ (np.abs(y+n).max()+1e-9)
def 时间平移(y,秒=0.8):  # 把声音在3秒里挪个位置（模拟鱼叫早一点/晚一点）
    sh=int(秒*SR); return np.roll(y,sh)
def 音量(y,k=0.4):       # 整体变小声（模拟水听器离鱼远/近）
    return y*k
# ---------- 频谱级增强 SpecAugment ----------
def 频率遮挡(mel,最大宽度=16):   # 随机抹掉一段"频率"（模拟某个频率被干扰盖住）
    M=mel.copy(); f=rng.integers(0,最大宽度); f0=rng.integers(0,N_MELS-f)
    M[f0:f0+f,:]=M.min(); return M
def 时间遮挡(mel,最大宽度=60):   # 随机抹掉一段"时间"（模拟有几帧没听到）
    M=mel.copy(); t=rng.integers(0,最大宽度); t0=rng.integers(0,N_FRAMES-t)
    M[:,t0:t0+t]=M.min(); return M
# ---------- Mixup ----------
def mixup(m1,m2,lam=0.7):        # 两张图按比例叠加，标签也按比例：0.7鱼+0.3别的
    return lam*m1+(1-lam)*m2

Mf=转mel(yf)
panels=[
 ("① 原始真鱼\n(基准)",Mf),
 ("② 加背景噪声\n(叠30%水流声)",转mel(加噪(yf,yw))),
 ("③ 时间平移\n(往后挪0.8秒)",转mel(时间平移(yf))),
 ("④ 音量变小\n(×0.4，模拟离得远)",转mel(音量(yf))),
 ("⑤ SpecAugment 频率遮挡\n(抹掉一段频率)",频率遮挡(Mf.copy())),
 ("⑥ SpecAugment 时间遮挡\n(抹掉几帧时间)",时间遮挡(Mf.copy())),
 ("⑦ Mixup 鱼+水流\n(0.7鱼+0.3水，软标签)",mixup(Mf,转mel(yw))),
 ("⑧ Mixup 鱼+应激\n(0.7鱼+0.3应激，软标签)",mixup(Mf,转mel(ys))),
]
fig,axes=plt.subplots(2,4,figsize=(16,7))
for ax,(title,M) in zip(axes.flat,panels):
    ax.imshow(M,origin="lower",aspect="auto",cmap="magma",extent=[0,3,0,14])
    ax.set_title(title,fontsize=11); ax.set_xlabel("时间(秒)"); ax.set_ylabel("频率(kHz)")
plt.suptitle("第18课a 数据增强：用一段真鱼变出8种略不同但还是鱼的训练样本",fontsize=13)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第18课a_数据增强长什么样.png",dpi=140,bbox_inches="tight")
print("图在 学习/第18课a_数据增强长什么样.png")
