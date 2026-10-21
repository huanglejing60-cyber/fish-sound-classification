# -*- coding: utf-8 -*-
"""第16课f：对f10/f13/f14做'成组咕咕'结构分析，并用已确认鱼f02/f12/f22做对照。
成组判定：间隔<150ms的脉冲归为同一'咕咕'（双/三脉冲），组间应有较长停顿。"""
import os, numpy as np, librosa, soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
D=r"E:\大黄鱼分析\dataset_train_v1\_待复核标签"; SR=32000
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
def 分组(pk,组内ms=150):
    if len(pk)<2:return [1],[]
    d=np.diff(pk)/SR*1000
    groups=[1]; grp_gap=[]
    for dd in d:
        if dd<组内ms: groups[-1]+=1
        else: groups.append(1); grp_gap.append(dd)
    return groups,d

files={"f10 存疑?":"f10_131505_train.wav","f13 存疑?":"f13_164612_train.wav","f14 存疑?":"f14_164651_train.wav",
       "f02 已确认鱼":"f02_040614_val.wav","f12 已确认鱼":"f12_164605_train.wav","f22 已确认鱼":"f22_222702_train.wav"}
fig,axes=plt.subplots(6,1,figsize=(13,11),sharex=True)
for ax,(name,fn) in zip(axes,files.items()):
    if not os.path.exists(os.path.join(D,fn)):
        # 兼容：文件名可能不同，找前缀
        pre=fn.split("_")[0]; cand=[x for x in os.listdir(D) if x.startswith(pre) and x.endswith(".wav") and "第" not in x]
        fn=cand[0]
    y=load(fn); yb,pk=鱼频段脉冲(y); groups,d=分组(pk)
    npair=sum(1 for g in groups if g>=2)
    t=np.arange(len(y))/SR
    ax.plot(t,yb,lw=.5,color="#6b7280")
    # 组成员用颜色标出
    idx=0; colors=plt.cm.tab10.colors
    for gi,g in enumerate(groups):
        c=colors[gi%10]
        for k in range(g):
            if idx<len(pk):
                ax.plot(pk[idx]/SR,yb[pk[idx]],"v",color=(c if g>=2 else "#374151"),ms=10 if g>=2 else 5)
                idx+=1
    密 = np.median(d) if len(d) else 0
    ax.set_title(f"{name}  脉冲{len(pk)}  成组(≥2声)数={npair}  各组大小={groups}  中位间隔{密:.0f}ms",fontsize=11)
    ax.set_ylabel("幅度")
axes[-1].set_xlabel("秒")
plt.tight_layout(); plt.savefig(os.path.join(D,"f10f13f14_存疑成组分析.png"),dpi=130,bbox_inches="tight")
print("存疑/对照的成组结构：")
for name,fn in files.items():
    if not os.path.exists(os.path.join(D,fn)):
        pre=fn.split("_")[0]; cand=[x for x in os.listdir(D) if x.startswith(pre) and x.endswith(".wav") and "第" not in x]; fn=cand[0]
    yb,pk=鱼频段脉冲(load(fn)); groups,d=分组(pk)
    npair=sum(1 for g in groups if g>=2)
    print(f"{name}: 脉冲{len(pk)} 成组数{npair} 各组{groups} 间隔(ms)={np.round(d).astype(int)}")
