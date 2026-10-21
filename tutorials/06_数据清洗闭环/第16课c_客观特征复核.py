# -*- coding: utf-8 -*-
"""第16课c：用客观特征复核6个耳听样本，尤其z03/z05是否真像鱼。
真鱼=鱼频段(700-1200Hz)脉冲多、间隔规律、谱质心低、背景干净。"""
import os, numpy as np, librosa, soundfile as sf
from scipy import signal
HEAR=r"E:\大黄鱼分析\dataset_train_v1\先听这6个"
SR=32000
files=["z00_基准真鱼_19点25.wav","z01_凌晨3点45.wav","z02_放鱼后10点04.wav",
       "z03_喂食16点46.wav","z04_人工种子17点41.wav","z05_夜间22点27.wav"]
def load(fn):
    y,sr=sf.read(os.path.join(HEAR,fn))
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    return y
def 特征(y):
    # 鱼频段带通
    b,a=signal.butter(4,[700/(SR/2),1200/(SR/2)],"band")
    yb=signal.filtfilt(b,a,y)
    env=np.abs(signal.hilbert(yb))                       # 包络
    win=int(0.005*SR); env=np.convolve(env,np.ones(win)/win,mode="same")
    med=np.median(env); mad=np.median(np.abs(env-med))+1e-9
    th=med+6*mad
    pk,_=signal.find_peaks(env,height=th,distance=int(0.02*SR))
    n=len(pk)
    间隔=np.diff(pk)/SR*1000
    cv=间隔.std()/间隔.mean() if len(间隔)>1 else np.nan  # 变异系数，越小越规律
    中位间隔=np.median(间隔) if len(间隔) else np.nan
    # 谱质心 & 峰频
    cent=librosa.feature.spectral_centroid(y=y,sr=SR,hop_length=512)[0].mean()
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512)); freqs=librosa.fft_frequencies(sr=SR,n_fft=2048)
    psd=(S**2).mean(1); pkf=freqs[np.argmax(psd[freqs<3000])]
    # 背景底噪：500Hz以下能量占比
    lo=psd[freqs<500].sum(); tot=psd.sum(); lowratio=lo/tot
    return n,中位间隔,cv,cent,pkf,lowratio
print(f"{'文件':<22}{'脉冲数':>6}{'中位间隔ms':>10}{'间隔CV':>8}{'谱质心Hz':>9}{'峰频Hz':>8}{'低频底噪%':>10}")
for fn in files:
    y=load(fn); n,gi,cv,cent,pkf,lr=特征(y)
    print(f"{fn:<22}{n:>6}{gi:>10.0f}{cv:>8.2f}{cent:>9.0f}{pkf:>8.0f}{lr*100:>10.0f}")
print("\n判读：真鱼应 脉冲数多、间隔CV小(规律)、谱质心低、低频底噪小")
