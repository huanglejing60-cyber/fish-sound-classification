# -*- coding: utf-8 -*-
# 探索：四道关检测器，跑一整段30分钟，看能否定位产卵高峰
import numpy as np, librosa
from scipy.signal import find_peaks, hilbert
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr=16000
y,_=librosa.load(wav,sr=sr,mono=True)
块=int(10*sr); n=len(y)//块
结果=[]
for bi in range(n):
    s=y[bi*块:(bi+1)*块]
    env=np.abs(hilbert(s)); k=int(0.005*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
    med=np.median(env); h=med+0.08*(env.max()-med)
    pk,_=find_peaks(env,distance=int(0.02*sr),height=h,prominence=0.02)
    if len(pk)==0: 结果.append((bi,0,0,0,0)); continue
    过关=[]; 峰频=[]
    for p in pk:
        a=max(0,p-int(0.008*sr)); b=p+int(0.008*sr)
        seg=s[a:b]*np.hanning(b-a)
        nfft=2048; A=np.abs(np.fft.rfft(seg,nfft)); f=np.fft.rfftfreq(nfft,1/sr)
        m=(f>=300)&(f<=1500); f,A=f[m],A[m]
        f0=f[np.argmax(A)]
        # 窄带占比：峰频±120Hz能量 / 总能量
        窄=np.sum(A[(f>=f0-120)&(f<=f0+120)])/np.sum(A)
        if 650<=f0<=1000 and 窄>0.5: 过关.append(p)
    # 成组：过关脉冲里间隔80-150ms的相邻对
    组数=0
    if len(过关)>=2:
        t=np.array(过关)/sr
        d=np.diff(t)*1000
        组数=int(np.sum((d>=80)&(d<=150)))
    判鱼=1 if 组数>=1 else 0
    结果.append((bi,len(pk),len(过关),组数,判鱼))
结果=np.array(结果)
# 打印每分钟的鱼声块数
print("块(10秒) | 候选数 | 过关数 | 组数 | 判鱼")
for bi in range(0,n,6):  # 每分钟打印
    seg=结果[bi:bi+6]
    print(f"第{bi//6:2d}分钟: 候选{int(seg[:,1].sum()):3d} 过关{int(seg[:,2].sum()):3d} 组数{int(seg[:,3].sum()):3d} 鱼声块{int(seg[:,4].sum())}/6")
print(f"\n总鱼声块: {int(结果[:,4].sum())}/{n}")
