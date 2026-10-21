# -*- coding: utf-8 -*-
"""
=====================================================================
第 06 课：给检测器打分——混淆矩阵 + 精度/召回/F1（机器学习评估核心）
---------------------------------------------------------------------
考试设置：
  卷子（标准答案）：165个3秒样本，你之前亲手耳听标注的
                   fish真鱼52 + stress应激26 + water水流46 + impact撞击41
  考生：规则检测器（规则：一块里有>=1对80~150ms成组脉冲 → 判为鱼声）
  真值：fish算"真鱼"，其余三类算"非鱼"
考完把"预测"和"标准答案"对比，算4个分数。
=====================================================================
"""
import numpy as np, librosa, glob, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

d = r"E:\大黄鱼分析\dataset_train_v1"; sr = 16000

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
    return 过窄,组

# 逐样本考试：真鱼=1，非鱼=0；规则：成组对>=1 → 判鱼
TP=FP=FN=TN=0                       # 四个计数器，先清零
四类误报={"stress":0,"water":0,"impact":0}
for c in ["fish","stress","water","impact"]:
    for f in glob.glob(os.path.join(d,c,"*.wav")):
        真鱼 = 1 if c=="fish" else 0
        过窄数, 组数 = 提特征(f)
        判鱼 = 1 if 过窄数>=1 and 组数>=1 else 0

        if 真鱼==1 and 判鱼==1: TP+=1      # 命中：是鱼、判成鱼
        elif 真鱼==1 and 判鱼==0: FN+=1    # 漏报：是鱼、没判出来
        elif 真鱼==0 and 判鱼==1: FP+=1; 四类误报[c]+=1   # 误报：不是鱼、判成鱼
        else: TN+=1                                   # 正确拒绝：不是鱼、也判非鱼

# 四个分数
准确率=(TP+TN)/(TP+FP+FN+TN)
精度=TP/(TP+FP)
召回=TP/(TP+FN)
F1=2*精度*召回/(精度+召回)

print("="*56)
print("【混淆矩阵】行=真实，列=电脑预测")
print(f"              预测=鱼   预测=非鱼")
print(f"  真鱼(52)      {TP:3d}      {FN:3d}")
print(f"  非鱼(113)     {FP:3d}      {TN:3d}")
print("="*56)
print(f"准确率 Accuracy = (TP+TN)/总数 = {准确率*100:.0f}%   （全部165个里判对的比例）")
print(f"精度 Precision  = TP/(TP+FP) = {精度*100:.0f}%   （报上来的里，有多少真鱼）")
print(f"召回 Recall     = TP/(TP+FN) = {召回*100:.0f}%   （52条真鱼，抓到了多少）")
print(f"F1 综合分       = 2PR/(P+R) = {F1*100:.0f}%   （精度和召回的综合）")
print("="*56)
print("误报（FP）来自哪些非鱼：", 四类误报, "→ 主要是应激声stress")

# 画图：左=2×2混淆矩阵，右=非鱼三类误报数
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={"width_ratios":[1,1.1]})
M=np.array([[TP,FN],[FP,TN]])
im=ax1.imshow(M,cmap="Blues",vmin=0,vmax=80)
for i in range(2):
    for j in range(2):
        ax1.text(j,i,str(M[i,j]),ha="center",va="center",fontsize=22,
                 color="white" if M[i,j]>45 else "black")
ax1.set_xticks([0,1]); ax1.set_xticklabels(["预测=鱼","预测=非鱼"])
ax1.set_yticks([0,1]); ax1.set_yticklabels(["真鱼(52)","非鱼(113)"])
ax1.set_title(f"混淆矩阵（规则检测器）\n精度{精度*100:.0f}% 召回{召回*100:.0f}% F1 {F1*100:.0f}%")
for r,c,t in [(0,0,"命中TP"),(0,1,"漏报FN"),(1,0,"误报FP"),(1,1,"正确拒绝TN")]:
    ax1.text(c,r+0.32,t,ha="center",fontsize=10,color="#555")
cats=list(四类误报); vals=[四类误报[k] for k in cats]
bars=ax2.bar(cats,vals,color=["#dc2626","#f59e0b","#f59e0b"])
for b,v in zip(bars,vals): ax2.text(b.get_x()+b.get_width()/2,v+0.4,str(v),ha="center",fontsize=13)
ax2.set_title("41个误报来自谁？——应激声stress被错判最多\n(应激声也'成组脉冲'，规则分不清咕咕和应激)")
ax2.set_ylabel("被误判成鱼的数量"); ax2.set_ylim(0,30)
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第06课_检测器打分_混淆矩阵.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("图已保存：",out)
