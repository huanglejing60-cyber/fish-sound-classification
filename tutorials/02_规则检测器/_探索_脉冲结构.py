# -*- coding: utf-8 -*-
import numpy as np, librosa
from scipy.signal import find_peaks, hilbert
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr=16000
y,_=librosa.load(wav,sr=sr,mono=True)
w=sr*10; n=len(y)//w
best,bk=-1,0
for i in range(n):
    s=y[i*w:(i+1)*w]; r=np.sqrt(np.mean(s**2))
    if r>best: best,bk=r,i
y10=y[bk*w:(bk+1)*w]
env=np.abs(hilbert(y10)); k=int(0.005*sr)
env=np.convolve(env,np.ones(k)/k,mode="same")
med=np.median(env); height=med+0.06*(env.max()-med)
peaks,_=find_peaks(env,distance=int(0.02*sr),height=height,prominence=0.02)
pt=peaks/sr
gaps=np.diff(pt)*1000
# 找连续3个间隔都在80-150ms的三脉冲组
for j in range(len(gaps)-2):
    if all(80<=gaps[j+m]<=150 for m in range(2)):
        print("三脉冲组起点(秒):",round(pt[j],3),"  三间隔(ms):",round(gaps[j]),round(gaps[j+1]))
        break
# 也列出所有脉冲时间，便于挑放大窗口
print("所有脉冲时间(秒):",np.round(pt,2))
