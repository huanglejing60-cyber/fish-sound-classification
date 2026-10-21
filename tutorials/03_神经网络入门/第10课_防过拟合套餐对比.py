# -*- coding: utf-8 -*-
"""
=====================================================================
第 10 课：治"过拟合"——数据增强+类别权重，看训练/验证差距怎么缩小
---------------------------------------------------------------------
同样网络、mini-batch、80轮、种子，只改"防过拟合套餐"开关：
  A 朴素：直接训（训练能背到100%，但和验证差距大、验证loss乱跳）
  B 套餐：+数据增强(背不住) +类别权重(照顾应激)
重点不是看谁最高，而是看"训练-验证差距(gap)"和"验证loss稳不稳"。
=====================================================================
"""
import os, csv
import numpy as np, librosa
import torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

ROOT=r"E:\大黄鱼分析\dataset_train_v1"; DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS=["fish","stress","water","impact"]; CN={"fish":"真鱼","stress":"麻醉应激","water":"水流设备","impact":"撞击"}
SR=32000; N_MELS=64; N_FRAMES=300; EPOCHS=80; BATCH=16

def 读mel(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=N_MELS,fmin=50,fmax=14000)
    S=librosa.power_to_db(S).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
def 组装(rs):
    X=np.stack([读mel(os.path.join(ROOT,r["类别"],r["文件名"])) for r in rs])[:,None]
    Y=np.array([int(r["类别id"]) for r in rs]); return torch.from_numpy(X).float(),torch.from_numpy(Y).long()
print("读取165个音频并转mel...")
Xtr,Ytr=组装(tr); Xva,Yva=组装(va)
mean,std=Xtr.mean(),Xtr.std()+1e-6; Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
Xtr,Ytr=Xtr.to(DEVICE),Ytr.to(DEVICE); Xva,Yva=Xva.to(DEVICE),Yva.to(DEVICE)

class 卷积块(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(),nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)
class 小CNN(nn.Module):
    def __init__(self,n=4):
        super().__init__()
        self.特征=nn.Sequential(卷积块(1,16),卷积块(16,32),卷积块(32,64),卷积块(64,128))
        self.池化=nn.AdaptiveAvgPool2d((1,1)); self.丢弃=nn.Dropout(0.3); self.分类=nn.Linear(128,n)
    def forward(self,x):
        x=self.特征(x); x=self.池化(x).flatten(1); x=self.丢弃(x); return self.分类(x)

def 增强(x,rng):
    if rng.random()<0.5: x=torch.roll(x,int(rng.integers(-20,21)),dims=1)
    x=x*float(rng.uniform(0.9,1.1))
    for _ in range(rng.integers(1,3)):
        w=int(rng.integers(2,9)); f0=int(rng.integers(0,N_MELS-w)); x[f0:f0+w,:]=0
    for _ in range(rng.integers(1,3)):
        w=int(rng.integers(5,31)); t0=int(rng.integers(0,N_FRAMES-w)); x[:,t0:t0+w]=0
    return x

cnt=np.array([(Ytr.cpu()==i).sum().item() for i in range(4)]); w=len(tr)/(4*np.maximum(cnt,1))

def 训练(用增强,用加权,标签):
    torch.manual_seed(42); np.random.seed(42)
    m=小CNN(4).to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEVICE) if 用加权 else None)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS) if 用增强 else None
    rng=np.random.default_rng(42); hist={"tr_acc":[],"va_acc":[],"va_loss":[]}; best=0; bestw=None
    for ep in range(1,EPOCHS+1):
        m.train(); idx=torch.randperm(len(tr)); 对=0
        for s in range(0,len(tr),BATCH):
            bi=idx[s:s+BATCH]; xb=Xtr[bi].clone(); yb=Ytr[bi]
            if 用增强:
                for i in range(len(xb)): xb[i,0]=增强(xb[i,0],rng)
            opt.zero_grad(); out=m(xb); l=crit(out,yb); l.backward(); opt.step()
            对+=(out.argmax(1)==yb).sum().item()
        if sched: sched.step()
        m.eval()
        with torch.no_grad():
            vo=m(Xva); vl=crit(vo,Yva).item(); va=(vo.argmax(1)==Yva).float().mean().item()
        hist["tr_acc"].append(对/len(tr)); hist["va_acc"].append(va); hist["va_loss"].append(vl)
        if va>best: best=va; bestw={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bestw); m.eval()
    with torch.no_grad(): pred=m(Xva).argmax(1).cpu().numpy()
    print(f"{标签}：最佳验证{best:.0%}，末轮训练{hist['tr_acc'][-1]:.0%}/验证{hist['va_acc'][-1]:.0%}，末轮训练-验证gap={hist['tr_acc'][-1]-hist['va_acc'][-1]:.0%}")
    return hist,best,pred

hA,bA,_=训练(False,False,"A 朴素")
hB,bB,predB=训练(True,True, "B 套餐")
真=Yva.cpu().numpy()

fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for ax0,h,t,c in [(ax[0],hA,"A 朴素：训练背到100%，验证忽高忽低、剧烈抖动","#dc2626"),
                  (ax[1],hB,"B 套餐：验证曲线平滑、loss稳定(不赌最高分)","#16a34a")]:
    ax0.plot(h["tr_acc"],ls="--",c=c,label="训练准确率")
    ax0.plot(h["va_acc"],c=c,label="验证准确率")
    ax0.set_title(t,fontsize=11); ax0.set_xlabel("轮次"); ax0.set_ylim(0,1.05); ax0.legend(); ax0.grid(alpha=.3)
ax[2].plot(hA["va_loss"],c="#dc2626",label="A 朴素：验证loss乱跳、反弹")
ax[2].plot(hB["va_loss"],c="#16a34a",label="B 套餐：验证loss平稳下降")
ax[2].set_title("验证loss：套餐后压得更稳更低",fontsize=11); ax[2].set_xlabel("轮次"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第10课_防过拟合对比.png",dpi=140,bbox_inches="tight"); plt.close()

cm=confusion_matrix(真,predB,labels=list(range(4))); cmn=cm/np.maximum(cm.sum(1,keepdims=True),1)
fig,a=plt.subplots(figsize=(6.5,5.5)); a.imshow(cmn,cmap="Greens",vmin=0,vmax=1)
a.set_xticks(range(4)); a.set_xticklabels([CN[l] for l in LABELS],rotation=30,ha="right")
a.set_yticks(range(4)); a.set_yticklabels([CN[l] for l in LABELS]); a.set_xlabel("预测"); a.set_ylabel("真实")
for i in range(4):
    for j in range(4): a.text(j,i,f"{cm[i,j]}\n{cmn[i,j]:.0%}",ha="center",va="center",color="white" if cmn[i,j]>0.5 else "black")
a.set_title(f"第10课 套餐版 验证集混淆矩阵（最佳{bB:.0%}）")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第10课_套餐版混淆矩阵.png",dpi=140,bbox_inches="tight")
print(f"\n图在 学习/ 目录。最高分 A {bA:.0%} vs B {bB:.0%}（40样本上差1个不算数，关键看gap和loss）")
