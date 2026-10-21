# -*- coding: utf-8 -*-
"""第16课d：对23个待复核批量提取客观特征，参照真鱼标准给机器建议，
把人工耳听工作量降到最低（只听'存疑'）。"""
import os, re, numpy as np, librosa, soundfile as sf
from scipy import signal
D=r"E:\大黄鱼分析\dataset_train_v1\_待复核标签"; SR=32000
def load(p):
    y,sr=sf.read(p)
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    return y
def 特征(y):
    b,a=signal.butter(4,[700/(SR/2),1200/(SR/2)],"band")
    yb=signal.filtfilt(b,a,y)
    env=np.abs(signal.hilbert(yb)); win=int(0.005*SR)
    env=np.convolve(env,np.ones(win)/win,mode="same")
    med=np.median(env);mad=np.median(np.abs(env-med))+1e-9
    pk,_=signal.find_peaks(env,height=med+6*mad,distance=int(0.02*SR))
    n=len(pk); g=np.diff(pk)/SR*1000
    cv=g.std()/g.mean() if len(g)>1 else 9
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512)); fq=librosa.fft_frequencies(sr=SR,n_fft=2048)
    psd=(S**2).mean(1); pkf=fq[np.argmax(psd[fq<3000])]
    lr=psd[fq<500].sum()/psd.sum()
    return n,cv,pkf,lr
def 建议(n,cv,pkf,lr):
    s=0
    if n>=15:s+=1
    if cv<=0.85:s+=1
    if lr<=0.05:s+=1
    if 500<=pkf<=1100:s+=1
    if s>=3:return "像鱼",s
    if s<=1:return "非鱼",s
    return "存疑",s
print(f"{'编号':<6}{'钟点':>9}{'脉冲':>5}{'CV':>6}{'峰频':>7}{'底噪%':>7}  建议")
res=[]
for fn in sorted(os.listdir(D)):
    if not fn.endswith(".wav"):continue
    m=re.match(r"(f\d+)_(\d{6})_",fn)
    tag=m.group(1); t=m.group(2); hm=f"{t[:2]}:{t[2:4]}"
    n,cv,pkf,lr=特征(load(os.path.join(D,fn)))
    adv,s=建议(n,cv,pkf,lr)
    res.append((tag,hm,n,cv,pkf,lr,adv))
    print(f"{tag:<6}{hm:>9}{n:>5}{cv:>6.2f}{pkf:>7.0f}{lr*100:>7.0f}  {adv}({s}/4)")
from collections import Counter
print("\n机器建议统计：",dict(Counter(r[6] for r in res)))
