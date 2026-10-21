# -*- coding: utf-8 -*-
# 对165个样本，用不同规则预测鱼/非鱼，计算精度/召回/F1（真值：fish=真鱼）
import numpy as np, librosa, glob, os
from scipy.signal import find_peaks, hilbert
d=r"E:\大黄鱼分析\dataset_train_v1"; sr=16000
def 特征(path):
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
    return 过窄,组
data=[]
for c in ["fish","stress","water","impact"]:
    for f in glob.glob(os.path.join(d,c,"*.wav")):
        n,z=特征(f); data.append((1 if c=="fish" else 0,n,z,c))
def 评分(规则名, fn):
    tp=fp=fn_=tn=0
    for 真鱼,n,z,c in data:
        判=1 if fn(n,z) else 0
        if 真鱼 and 判: tp+=1
        elif 真鱼 and not 判: fn_+=1
        elif not 真鱼 and 判: fp+=1
        else: tn+=1
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn_) if tp+fn_ else 0
    F=2*P*R/(P+R) if P+R else 0; acc=(tp+tn)/len(data)
    print(f"{规则名:18s} TP={tp:2d} FP={fp:2d} FN={fn_:2d} TN={tn:2d} | 精度{P*100:4.0f}% 召回{R*100:4.0f}% F1 {F*100:4.0f}% 准确率{acc*100:4.0f}%")
评分("成组对>=1", lambda n,z: z>=1)
评分("过窄脉冲>=1", lambda n,z: n>=1)
评分("过窄>=1且成组>=1", lambda n,z: n>=1 and z>=1)
评分("过窄>=2", lambda n,z: n>=2)
# 看各非鱼类被规则误报的情况（用"过窄>=1"）
print("\n各非鱼类被'过窄脉冲>=1'误报的数量:")
for c in ["stress","water","impact"]:
    误=sum(1 for 真,n,z,cc in data if cc==c and n>=1)
    print(f"  {c}: {误}")
