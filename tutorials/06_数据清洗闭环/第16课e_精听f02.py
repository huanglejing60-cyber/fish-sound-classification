# -*- coding: utf-8 -*-
"""第16课e：把存疑的f02切成1秒片段并画精听图，定位到底有没有鱼声。"""
import os, numpy as np, librosa, soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
D=r"E:\大黄鱼分析\dataset_train_v1\_待复核标签"
SR=32000
def load(fn):
    y,sr=sf.read(os.path.join(D,fn))
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    return y
def 带通(y):
    b,a=signal.butter(4,[700/(SR/2),1200/(SR/2)],"band")
    return signal.filtfilt(b,a,y)

f02=load("f02_040614_val.wav"); z00=load("f22_222702_train.wav")  # f22即z05已确认的鱼，作参照
# 切3个1秒片段，导出便于逐秒听
for i in range(3):
    seg=f02[i*SR:(i+1)*SR]
    sf.write(os.path.join(D,f"f02_第{i+1}秒.wav"),seg,SR)
print("已切出 f02_第1秒/第2秒/第3秒.wav")

def 脉冲位置(y):
    yb=带通(y); env=np.abs(signal.hilbert(yb))
    w=int(0.005*SR); env=np.convolve(env,np.ones(w)/w,mode="same")
    med=np.median(env);mad=np.median(np.abs(env-med))+1e-9
    pk,_=signal.find_peaks(env,height=med+6*mad,distance=int(0.02*SR))
    return yb,pk,env

fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=False)
# 语谱
M=librosa.feature.melspectrogram(y=f02,sr=SR,n_fft=1024,hop_length=320,n_mels=64,fmin=50,fmax=14000)
axes[0].imshow(librosa.power_to_db(M,ref=1.0,amin=1e-10),origin="lower",aspect="auto",cmap="magma",extent=[0,3,50,14000])
axes[0].set_title("f02（3.14凌晨4:06）语谱  ——  红线标出鱼频段(700-1200Hz)脉冲")
axes[0].set_ylabel("Hz"); axes[0].axhline(700,color="lime",ls="--",lw=.7); axes[0].axhline(1200,color="lime",ls="--",lw=.7)
yb,pk,env=脉冲位置(f02)
t=np.arange(len(f02))/SR
axes[1].plot(t,yb,lw=.5,color="#374151")
axes[1].plot(pk/SR,yb[pk],"v",color="#dc2626",ms=9)
for k,p in enumerate(pk): axes[1].annotate(str(k+1),(p/SR,yb[p]),fontsize=8,color="#dc2626",xytext=(0,6),textcoords="offset points",ha="center")
axes[1].set_title(f"f02 鱼频段波形（检测到{len(pk)}个脉冲，看间距规不规律）");axes[1].set_ylabel("幅度")
yb0,pk0,_=脉冲位置(z00)
axes[2].plot(np.arange(len(z00))/SR,yb0,lw=.5,color="#15803d")
axes[2].plot(pk0/SR,yb0[pk0],"v",color="#15803d",ms=7)
axes[2].set_title(f"对照：已确认真鱼f22(z05)的鱼频段（{len(pk0)}个脉冲，规律成组）");axes[2].set_ylabel("幅度");axes[2].set_xlabel("秒")
plt.tight_layout(); plt.savefig(os.path.join(D,"f02_精听图.png"),dpi=130,bbox_inches="tight")
print("脉冲时刻(ms):",[int(p/SR*1000) for p in pk])
