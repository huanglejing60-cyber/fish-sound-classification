# -*- coding: utf-8 -*-
"""
=====================================================================
第12课b：宁漏勿错——给模型加"置信度门槛"，把精度拉到100%
---------------------------------------------------------------------
导师要求：抓到的鱼必须是鱼(精度拉满，误报=0)；可以漏掉一部分(召回可以低)。
做法：不用argmax硬选，改成"鱼概率 ≥ 门槛T 才报鱼，否则不报"。
扫描不同门槛T，看精度/召回/误报数，找出【误报=0】的最高召回工作点。
=====================================================================
"""
import os, csv
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL=r"E:\大黄鱼分析\models\cnn14_finetune.pt"
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_layer(l):
    if isinstance(l,(nn.Linear,nn.Conv2d)):
        nn.init.xavier_uniform_(l.weight)
        if l.bias is not None: l.bias.data.fill_(0.)
def init_bn(bn): bn.bias.data.fill_(0.); bn.weight.data.fill_(1.)
class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.conv1=nn.Conv2d(ci,co,3,padding=1,bias=False); self.conv2=nn.Conv2d(co,co,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(co); self.bn2=nn.BatchNorm2d(co)
        init_layer(self.conv1); init_layer(self.conv2); init_bn(self.bn1); init_bn(self.bn2)
    def forward(self,x):
        x=F.relu_(self.bn1(self.conv1(x))); x=F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x,kernel_size=(2,2),stride=(2,2))
class Cnn14(nn.Module):
    def __init__(self,classes=4):
        super().__init__()
        self.bn0=nn.BatchNorm2d(N_MELS)
        self.conv_block1=ConvBlock(1,64); self.conv_block2=ConvBlock(64,128); self.conv_block3=ConvBlock(128,256)
        self.conv_block4=ConvBlock(256,512); self.conv_block5=ConvBlock(512,1024); self.conv_block6=ConvBlock(1024,2048)
        self.fc1=nn.Linear(2048,2048); self.fc_audioset=nn.Linear(2048,classes)
        init_bn(self.bn0); init_layer(self.fc1); init_layer(self.fc_audioset)
    def forward(self,x):
        x=x.unsqueeze(1).permute(0,2,3,1); x=self.bn0(x); x=x.permute(0,3,2,1)
        x=self.conv_block1(x); x=self.conv_block2(x); x=self.conv_block3(x)
        x=self.conv_block4(x); x=self.conv_block5(x); x=self.conv_block6(x)
        x=torch.mean(x,dim=3); x1,_=torch.max(x,dim=2); x=x1+torch.mean(x,dim=2)
        x=F.dropout(x,0.5,self.training); x=F.relu_(self.fc1(x)); emb=F.dropout(x,0.5,self.training)
        return self.fc_audioset(emb),emb

ckpt=torch.load(MODEL,map_location=DEVICE,weights_only=False)
labels=ckpt["labels"]; fish_idx=labels.index("fish"); mean=float(ckpt["mean"]); std=float(ckpt["std"]+1e-6)
模型=Cnn14(4); 模型.load_state_dict(ckpt["state_dict"]); 模型.to(DEVICE).eval()

def 读mel(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
def 鱼概率(path):
    x=(读mel(path)-mean)/std; x=torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p=torch.softmax(模型(x)[0],dim=1)[0].cpu().numpy()
    return float(p[fish_idx])

rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
va=[r for r in rows if r["split"]=="val"]
数据=[]
for r in va:
    数据.append((鱼概率(os.path.join(ROOT,r["类别"],r["文件名"])), 1 if r["类别"]=="fish" else 0))
真鱼数=sum(y for _,y in 数据); 非鱼数=len(数据)-真鱼数
print(f"验证集40个：真鱼{真鱼数}个，非鱼{非鱼数}个\n")
print("门槛T | 报鱼数 | 真抓到 误报 漏掉 | 精度   召回")
print("-"*58)
Ts=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
P=[]; R=[]; N报=[]
for T in Ts:
    TP=FP=FN=0
    for pf,y in 数据:
        报= pf>=T
        if y and 报: TP+=1
        elif (not y) and 报: FP+=1
        elif y and not 报: FN+=1
    prec=TP/(TP+FP) if TP+FP>0 else 1.0
    rec=TP/真鱼数
    P.append(prec); R.append(rec); N报.append(TP+FP)
    print(f"{T:4.2f} | {TP+FP:5d} | {TP:5d} {FP:4d} {FN:4d} | {prec:5.0%} {rec:5.0%}")
# 找误报=0里召回最高的门槛
零误=[(T,r) for T,r,p in zip(Ts,R,P) if p>=0.999]
推荐=max(零误,key=lambda z:z[1]) if 零误 else None
print("-"*58)
if 推荐: print(f"推荐工作点：门槛 T={推荐[0]}（此时误报=0、精度100%，召回{推荐[1]:.0%}，完全符合导师要求）")

fig,ax=plt.subplots(1,2,figsize=(12.5,4.8))
ax[0].plot(Ts,P,"o-",c="#2563eb",label="精度（抓到的几成是真鱼）")
ax[0].plot(Ts,R,"s-",c="#16a34a",label="召回（真鱼抓到几成）")
ax[0].set_xlabel("鱼概率门槛 T（越高越严格）"); ax[0].set_ylim(0,1.05); ax[0].grid(alpha=.3); ax[0].legend()
ax[0].set_title("门槛越高：精度↑ 召回↓（跷跷板）")
if 推荐:
    ax[0].axvline(推荐[0],ls="--",c="#dc2626"); ax[0].annotate(f"推荐 T={推荐[0]}\n误报0",(推荐[0],0.15),color="#dc2626",ha="right")
# 右图：每条鱼/非鱼的鱼概率分布
鱼p=[pf for pf,y in 数据 if y==1]; 非p=[pf for pf,y in 数据 if y==0]
ax[1].scatter(鱼p,np.random.uniform(0.6,1.0,len(鱼p)),c="#16a34a",s=45,label="真鱼",alpha=.8)
ax[1].scatter(非p,np.random.uniform(0.0,0.4,len(非p)),c="#dc2626",s=45,label="非鱼",alpha=.8,marker="x")
if 推荐: ax[1].axvline(推荐[0],ls="--",c="#dc2626",label=f"门槛{推荐[0]}：线右边才报鱼")
ax[1].set_xlabel("模型给的鱼概率"); ax[1].set_yticks([]); ax[1].set_title("每个样本的鱼概率：门槛右边的才报")
ax[1].legend(); ax[1].grid(alpha=.3,axis="x")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第12课b_宁漏勿错_阈值扫描.png",dpi=140,bbox_inches="tight")
print("\n图在 学习/第12课b_宁漏勿错_阈值扫描.png")
