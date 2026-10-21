# -*- coding: utf-8 -*-
# 探索：四类样本在检测器特征下的分布，为"考试"定判定规则
import numpy as np, librosa, glob, os
from scipy.signal import find_peaks, hilbert
d=r"E:\大黄鱼分析\dataset_train_v1"
sr=16000
def 特征(path):
    y,_=librosa.load(path,sr=sr,mono=True)
    env=np.abs(hilbert(y)); k=int(0.005*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
    med=np.median(env); h=med+0.08*(env.max()-med)
    pk,_=find_peaks(env,distance=int(0.02*sr),height=h,prominence=0.02)
    过窄=0
    for p in pk:
        a=max(0,p-int(0.008*sr)); b=p+int(0.008*sr)
        seg=y[a:b]; seg=seg*np.hanning(len(seg)); A=np.abs(np.fft.rfft(seg,2048)); f=np.fft.rfftfreq(2048,1/sr)
        m=(f>=300)&(f<=1500); f,A=f[m],A[m]
        if len(A)==0: continue
        f0=f[np.argmax(A)]
        if 650<=f0<=1000 and np.sum(A[(f>=f0-120)&(f<=f0+120)])/np.sum(A)>0.5: 过窄+=1
    组=0
    if len(pk)>=2:
        dd=np.diff(pk/sr)*1000; 组=int(np.sum((dd>=80)&(dd<=150)))
    return len(pk),过窄,组
for c in ["fish","stress","water","impact"]:
    files=glob.glob(os.path.join(d,c,"*.wav"))
    脉冲=[];过窄=[];组=[]
    for f in files:
        p,n,z=特征(f); 脉冲.append(p);过窄.append(n);组.append(z)
    脉冲,过窄,组=map(np.array,(脉冲,过窄,组))
    print(f"{c:7s} n={len(files):2d} | 脉冲中位{np.median(脉冲):.0f} 过窄脉冲中位{np.median(过窄):.0f} "
          f"成组对中位{np.median(组):.0f} | 过窄>=1的:{np.sum(过窄>=1):2d}  过窄>=2:{np.sum(过窄>=2):2d}  有成组对:{np.sum(组>=1):2d}")
