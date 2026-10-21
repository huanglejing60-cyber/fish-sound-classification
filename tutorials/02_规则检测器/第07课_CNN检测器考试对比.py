# -*- coding: utf-8 -*-
"""
=====================================================================
第 07 课：CNN14 和规则检测器，考同一张卷子（验证集40个，模型没见过）
---------------------------------------------------------------------
公平原则：只用 split=="val" 的40个样本（训练时特意留出、模型没学过），
        让规则法和CNN都来判，各算各的混淆矩阵、精度、召回、F1，
        亲眼看看深度学习比人工规则强多少。
=====================================================================
"""
import os, csv
import numpy as np, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
import torch, torch.nn as nn, torch.nn.functional as F
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

ROOT=r"E:\大黄鱼分析\dataset_train_v1"; PT=r"E:\大黄鱼分析\models\cnn14_finetune.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 一、规则检测器（第5-6课的成组规则，在16k上做）----------
def 规则判鱼(path):
    sr=16000
    y,_=librosa.load(path,sr=sr,mono=True)
    env=np.abs(hilbert(y)); k=int(0.005*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
    med=np.median(env); h=med+0.08*(env.max()-med)
    pk,_=find_peaks(env,distance=int(0.02*sr),height=h,prominence=0.02)
    组=0
    if len(pk)>=2:
        d=np.diff(pk/sr)*1000; 组=int(np.sum((d>=80)&(d<=150)))
    return 1 if 组>=1 else 0

# ---------- 二、CNN14模型结构（这段是模型定义，会用即可，不用现在懂）----------
def init_layer(l):
    if isinstance(l,(nn.Linear,nn.Conv2d)):
        nn.init.xavier_uniform_(l.weight)
        if l.bias is not None: l.bias.data.fill_(0.)
class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.conv1=nn.Conv2d(ci,co,3,padding=1,bias=False); self.conv2=nn.Conv2d(co,co,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(co); self.bn2=nn.BatchNorm2d(co)
    def forward(self,x):
        x=F.relu_(self.bn1(self.conv1(x))); x=F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x,2,2)
class Cnn14(nn.Module):
    def __init__(self,classes_num=4):
        super().__init__()
        N_MELS=64
        self.bn0=nn.BatchNorm2d(N_MELS)
        self.conv_block1=ConvBlock(1,64); self.conv_block2=ConvBlock(64,128); self.conv_block3=ConvBlock(128,256)
        self.conv_block4=ConvBlock(256,512); self.conv_block5=ConvBlock(512,1024); self.conv_block6=ConvBlock(1024,2048)
        self.fc1=nn.Linear(2048,2048); self.fc_audioset=nn.Linear(2048,classes_num)
    def forward(self,x):
        x=x.unsqueeze(1).permute(0,2,3,1); x=self.bn0(x); x=x.permute(0,3,2,1)
        for b in [self.conv_block1,self.conv_block2,self.conv_block3,self.conv_block4,self.conv_block5,self.conv_block6]:
            x=b(x)
        x=torch.mean(x,dim=3); x1,_=torch.max(x,dim=2); x=x1+torch.mean(x,dim=2)
        x=F.dropout(x,0.5,self.training); x=F.relu_(self.fc1(x)); x=F.dropout(x,0.5,self.training)
        return self.fc_audioset(x)

# 读验证集
rows=[]
with open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["split"]=="val":
            rows.append(r)
print(f"验证集 {len(rows)} 个，设备 {DEVICE}")

# 加载训练好的模型
ck=torch.load(PT,map_location=DEVICE,weights_only=False)
model=Cnn14(4); model.load_state_dict(ck["state_dict"]); model.to(DEVICE).eval()
mean,std,labels=ck["mean"],ck["std"],ck["labels"]
print("类别顺序:",labels,"（id=0是fish）")

def cnn判鱼(path):
    SR=32000
    y,_=librosa.load(path,sr=SR,mono=True)
    y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=64,fmin=50,fmax=14000)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    S=np.pad(S,((0,0),(0,max(0,300-S.shape[1]))))[:,:300]
    X=torch.from_numpy((S-mean)/std).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits=model(X)
    return int(logits.argmax(1).item()==0)   # 预测成fish(0)就是判鱼

# 逐样本考试
def 打分(预测表):
    tp=fp=fn=tn=0
    for 真鱼,判 in 预测表:
        if 真鱼 and 判: tp+=1
        elif 真鱼 and not 判: fn+=1
        elif not 真鱼 and 判: fp+=1
        else: tn+=1
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F=2*P*R/(P+R) if P+R else 0
    return np.array([[tp,fn],[fp,tn]]),P,R,F
规则表=[]; cnn表=[]
for r in rows:
    path=os.path.join(ROOT,r["类别"],r["文件名"]); 真鱼=1 if r["类别"]=="fish" else 0
    规则表.append((真鱼,规则判鱼(path))); cnn表.append((真鱼,cnn判鱼(path)))
cmr,Pr,Rr,Fr=打分(规则表); cmc,Pc,Rc,Fc=打分(cnn表)

print("\n=== 规则检测器（验证集）===")
print(f"TP{cmr[0,0]} FN{cmr[0,1]} FP{cmr[1,0]} TN{cmr[1,1]} | 精度{Pr*100:.0f}% 召回{Rr*100:.0f}% F1 {Fr*100:.0f}%")
print("=== CNN14（验证集）===")
print(f"TP{cmc[0,0]} FN{cmc[0,1]} FP{cmc[1,0]} TN{cmc[1,1]} | 精度{Pc*100:.0f}% 召回{Rc*100:.0f}% F1 {Fc*100:.0f}%")

# 画图
fig,axs=plt.subplots(1,3,figsize=(15,4.6))
for ax,cm,title,P,R,F in [(axs[0],cmr,"规则检测器",Pr,Rr,Fr),(axs[1],cmc,"CNN14深度学习",Pc,Rc,Fc)]:
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=11)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=24,color="white" if cm[i,j]>6 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["预测鱼","预测非鱼"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["真鱼","非鱼"])
    for rr,cc,t in [(0,0,"TP"),(0,1,"FN"),(1,0,"FP"),(1,1,"TN")]:
        ax.text(cc,rr+0.3,t,ha="center",fontsize=10,color="#555")
    ax.set_title(f"{title}\n精度{P*100:.0f}% 召回{R*100:.0f}% F1 {F*100:.0f}%",fontsize=12)
x=np.arange(3); w=0.35
axs[2].bar(x-w/2,[Pr*100,Rr*100,Fr*100],w,label="规则",color="#94a3b8")
axs[2].bar(x+w/2,[Pc*100,Rc*100,Fc*100],w,label="CNN14",color="#16a34a")
for i,(a,b) in enumerate(zip([Pr*100,Rr*100,Fr*100],[Pc*100,Rc*100,Fc*100])):
    axs[2].text(i-w/2,a+1,f"{a:.0f}",ha="center"); axs[2].text(i+w/2,b+1,f"{b:.0f}",ha="center")
axs[2].set_xticks(x); axs[2].set_xticklabels(["精度","召回","F1"]); axs[2].set_ylim(0,108)
axs[2].legend(); axs[2].set_title("同一张卷子：规则 vs CNN",fontsize=12); axs[2].grid(alpha=.3,axis="y")
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第07课_规则vsCNN_同卷对比.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("图已保存：",out)
