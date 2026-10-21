# -*- coding: utf-8 -*-
# 第6课练习：规则改成"过窄脉冲>=1 且 成组对>=1"（两个条件都满足才算鱼）
import numpy as np, librosa, glob, os
from scipy.signal import find_peaks, hilbert
d=r"E:\大黄鱼分析\dataset_train_v1"; sr=16000
def 提特征(path):
    y,_=librosa.load(path,sr=sr,mono=True)
    env=np.abs(hilbert(y)); k=int(0.005*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
    med=np.median(env); h=med+0.08*(env.max()-med)
    pk,_=find_peaks(env,distance=int(0.02*sr),height=h,prominence=0.02)
    过窄=0
    for p in pk:
        a=max(0,p-int(0.008*sr)); b=p+int(0.008*sr); seg=y[a:b]; seg=seg*np.hanning(len(seg))
        A=np.abs(np.fft.rfft(seg,2048)); f=np.fft.rfftfreq(2048,1/sr)
        m=(f>=300)&(f<=1500); f,A=f[m],A[m]
        if len(A)==0: continue
        f0=f[np.argmax(A)]
        if 650<=f0<=1000 and np.sum(A[(f>=f0-120)&(f<=f0+120)])/np.sum(A)>0.5: 过窄+=1
    组=0
    if len(pk)>=2:
        dd=np.diff(pk/sr)*1000; 组=int(np.sum((dd>=80)&(dd<=150)))
    return 过窄, 组          # ★改动1：函数返回两个值
TP=FP=FN=TN=0
for c in ["fish","stress","water","impact"]:
    for f in glob.glob(os.path.join(d,c,"*.wav")):
        真鱼=1 if c=="fish" else 0
        过窄数,组数=提特征(f)   # ★改动2：一次接住两个值
        判鱼=1 if 过窄数>=1 and 组数>=1 else 0
        if 真鱼 and 判鱼: TP+=1
        elif 真鱼 and not 判鱼: FN+=1
        elif not 真鱼 and 判鱼: FP+=1
        else: TN+=1
P=TP/(TP+FP); R=TP/(TP+FN); F=2*P*R/(P+R)
print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"精度{P*100:.0f}% 召回{R*100:.0f}% F1 {F*100:.0f}% 准确率{(TP+TN)/165*100:.0f}%")
