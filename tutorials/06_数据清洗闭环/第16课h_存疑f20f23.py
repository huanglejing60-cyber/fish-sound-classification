# -*- coding: utf-8 -*-
"""第16课h：对f20、f23做精细定性（成组结构+每脉冲中心频率），对照f22真鱼。"""
import os, numpy as np, librosa, soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
D=r"E:\大黄鱼分析\dataset_train_v1\_待复核标签"; SR=32000
def find(prefix):
    return [x for x in os.listdir(D) if x.startswith(prefix) and x.endswith(".wav") and "第" not in x and "段" not in x][0]
def load(fn):
    y,sr=sf.read(os.path.join(D,fn))
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    return y
def 鱼频段脉冲(y):
    b,a=signal.butter(4,[700/(SR/2),1200/(SR/2)],"band")
    yb=signal.filtfilt(b,a,y)
    env=np.abs(signal.hilbert(yb)); w=int(0.005*SR); env=np.convolve(env,np.ones(w)/w,mode="same")
    med=np.median(env);mad=np.median(np.abs(env-med))+1e-9
    pk,_=signal.find_peaks(env,height=med+6*mad,distance=int(0.02*SR))
    return yb,pk
def 分组(pk,ms=150):
    if len(pk)<2:return [1],[]
    d=np.diff(pk)/SR*1000; groups=[1]
    for dd in d:
        if dd<ms:groups[-1]+=1
        else:groups.append(1)
    return groups,d
def 脉冲中心频率(y,p):
    seg=y[max(0,p-int(.008*SR)):p+int(.008*SR)]
    if len(seg)<32:return 0,0
    S=np.abs(librosa.stft(seg,n_fft=512,hop_length=128)); fq=librosa.fft_frequencies(sr=SR,n_fft=512)
    psd=(S**2).mean(1); band=(fq>=500)&(fq<=1300)
    pkf=fq[band][np.argmax(psd[band])]
    cent=(fq*psd).sum()/psd.sum()
    return pkf,cent

files={"f20":"f20","f23":"f23","f22(已确认鱼)":"f22"}
fig,axes=plt.subplots(3,1,figsize=(13,8))
for ax,(name,pre) in zip(axes,files.items()):
    y=load(find(pre)); yb,pk=鱼频段脉冲(y); groups,d=分组(pk)
    freqs=[]
    for p in pk:
        pf,ce=脉冲中心频率(y,p); freqs.append(pf)
    M=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=64,fmin=50,fmax=14000)
    ax.imshow(librosa.power_to_db(M,ref=1.0,amin=1e-10),origin="lower",aspect="auto",cmap="magma",extent=[0,3,50,14000])
    ax.axhline(700,color="lime",ls="--",lw=.6);ax.axhline(1200,color="lime",ls="--",lw=.6)
    ax.scatter(pk/SR,np.full(len(pk),900),marker="v",color="#fff",s=60,edgecolor="#000")
    npair=sum(1 for g in groups if g>=2)
    ax.set_title(f"{name}: 脉冲{len(pk)} 成组数{npair} 各组{groups} 各脉冲峰频Hz={np.round(freqs).astype(int)}",fontsize=10)
    ax.set_ylabel("Hz")
plt.tight_layout(); plt.savefig(os.path.join(D,"f20f23_存疑定性.png"),dpi=130,bbox_inches="tight")
for name,pre in files.items():
    y=load(find(pre)); yb,pk=鱼频段脉冲(y); groups,d=分组(pk)
    freqs=[脉冲中心频率(y,p)[0] for p in pk]
    print(f"{name}: 脉冲{len(pk)} 各组{groups} 峰频={np.round(freqs).astype(int)} 间隔={np.round(d).astype(int)}")
