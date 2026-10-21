# -*- coding: utf-8 -*-
"""
=====================================================================
第 18 课 c：让你亲眼看到"数据增强到底在干嘛"——过拟合gap
---------------------------------------------------------------------
每轮同时记录两个分数：
  训练集准确率 = 做"练过的原题"（32道死背题）
  验证集准确率 = 做"没见过的新题"（真本事）
不增强：原题很快满分(死背)，新题上不去 → 两条线差距大(过拟合)
增强：  原题做不到满分(背不下来)，但它在学规律 → 两条线差距小
=====================================================================
"""
import os, csv
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL_DIR=r"E:\大黄鱼分析\models"
PRETRAIN=os.path.join(MODEL_DIR,"cnn14_hf.safetensors")
LABELS=["fish","stress","water","impact"]
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); BATCH=4; ACCUM=2; EPOCHS=25

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
def 借耳朵():
    from safetensors.torch import load_file
    raw=load_file(PRETRAIN); sd={}
    for k,v in raw.items():
        if k.startswith("backbone."):
            nk=k[len("backbone."):]
            if not(nk.startswith("spectrogram_extractor") or nk.startswith("logmel_extractor")): sd[nk]=v
    m=Cnn14(527); m.load_state_dict(sd,strict=True)
    m.fc_audioset=nn.Linear(2048,4); init_layer(m.fc_audioset); return m
def 读波形(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]; return y
def 转mel(y):
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
波tr=[读波形(os.path.join(ROOT,r["类别"],r["文件名"])) for r in tr]; Ytr=np.array([int(r["类别id"]) for r in tr])
Mva=np.stack([转mel(读波形(os.path.join(ROOT,r["类别"],r["文件名"]))) for r in va]); Yva=np.array([int(r["类别id"]) for r in va])
噪声池=[读波形(os.path.join(ROOT,r["类别"],r["文件名"])) for r in tr if r["类别"] in ("water","impact")]
Mtr0=np.stack([转mel(y) for y in 波tr]); mean,std=Mtr0.mean(),Mtr0.std()+1e-6
Xtr=torch.from_numpy((Mtr0-mean)/std).float().to(DEVICE); Ytr_t=torch.from_numpy(Ytr).long().to(DEVICE)
Xva=torch.from_numpy((Mva-mean)/std).float().to(DEVICE); Yva_t=torch.from_numpy(Yva).long().to(DEVICE)
cnt=np.array([(Ytr==i).sum() for i in range(4)]); w=len(tr)/(4*np.maximum(cnt,1))
rng=np.random.default_rng(0)
def 波形增强(y):
    y=y.copy()*rng.uniform(0.6,1.2); sh=int(rng.integers(-int(0.4*SR),int(0.4*SR))); y=np.roll(y,sh)
    if rng.random()<0.5:
        n=噪声池[rng.integers(0,len(噪声池))]; y=y+rng.uniform(0.1,0.4)*n
    return y
def specaug(M):
    M=M.copy()
    for _ in range(int(rng.integers(1,3))):
        f=int(rng.integers(0,12)); f0=int(rng.integers(0,max(1,N_MELS-f))); M[f0:f0+f,:]=M.min()
    for _ in range(int(rng.integers(1,3))):
        t=int(rng.integers(0,50)); t0=int(rng.integers(0,max(1,N_FRAMES-t))); M[:,t0:t0+t]=M.min()
    return M
def 增强epoch():
    X=np.stack([specaug(转mel(波形增强(y))) for y in 波tr])
    return torch.from_numpy((X-mean)/std).float().to(DEVICE)
def 打分(mm,X,Y):
    pp=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH):
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"): lo,_=mm(X[i:i+BATCH])
            pp.append(torch.softmax(lo,1).float().argmax(1))
    return (torch.cat(pp)==Y).float().mean().item()

def 训练(用增强,标签):
    torch.manual_seed(42); np.random.seed(42); rng=np.random.default_rng(123)
    m=借耳朵()
    for p in m.parameters(): p.requires_grad=False
    for i in [5,6]:
        for p in getattr(m,f"conv_block{i}").parameters(): p.requires_grad=True
    for p in m.fc1.parameters(): p.requires_grad=True
    for p in m.fc_audioset.parameters(): p.requires_grad=True
    冻=["conv_block1","conv_block2","conv_block3","conv_block4","bn0"]; m.to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEVICE))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    H={"tr":[],"va":[]}
    for ep in range(1,EPOCHS+1):
        m.train()
        for name,mod in m.named_modules():
            if isinstance(mod,nn.BatchNorm2d) and name.split(".")[0] in 冻: mod.eval()
        Xt=增强epoch() if 用增强 else Xtr
        idx=torch.randperm(len(tr)); opt.zero_grad()
        for bi0 in range(0,len(tr),BATCH):
            bi=idx[bi0:bi0+BATCH]
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
                out,_=m(Xt[bi]); loss=crit(out,Ytr_t[bi])/ACCUM
            scaler.scale(loss).backward()
            if (bi0//BATCH+1)%ACCUM==0 or (bi0+BATCH)>=len(tr):
                scaler.step(opt); scaler.update(); opt.zero_grad()
        m.eval(); tr_acc=打分(m,Xtr,Ytr_t); va_acc=打分(m,Xva,Yva_t)
        H["tr"].append(tr_acc); H["va"].append(va_acc)
        print(f"[{标签}] {ep:02d}轮 原题(训练){tr_acc:.0%} 新题(验证){va_acc:.0%} gap={tr_acc-va_acc:.0%}",flush=True)
    return H

print("===== 不增强：死背原题 =====",flush=True)
H0=训练(False,"无增强")
print("\n===== 增强：逼它学规律 =====",flush=True)
H1=训练(True,"增强")

fig,ax=plt.subplots(1,2,figsize=(13,5),sharey=True)
for a,H,t in [(ax[0],H0,"不增强：训练(原题)很快满分，验证(新题)上不去 → 过拟合"),
              (ax[1],H1,"增强：训练(原题)满不了分，因为它没背、在学规律")]:
    a.plot(H["tr"],c="#dc2626",lw=2,label="训练集=做原题")
    a.plot(H["va"],c="#2563eb",lw=2,label="验证集=做新题")
    a.fill_between(range(EPOCHS),H["va"],H["tr"],color="#fca5a5",alpha=.25)
    a.set_title(t,fontsize=11); a.set_ylim(0,1.02); a.set_xlabel("轮次"); a.legend(); a.grid(alpha=.3)
ax[0].set_ylabel("准确率")
plt.suptitle("第18课c 数据增强的作用：缩小'背原题'和'做新题'的差距（过拟合gap）",fontsize=13)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第18课c_过拟合gap.png",dpi=140,bbox_inches="tight")
print(f"\n最终：不增强 训练{H0['tr'][-1]:.0%}/验证{H0['va'][-1]:.0%} gap={H0['tr'][-1]-H0['va'][-1]:.0%}")
print(f"      增强   训练{H1['tr'][-1]:.0%}/验证{H1['va'][-1]:.0%} gap={H1['tr'][-1]-H1['va'][-1]:.0%}")
print("图在 学习/第18课c_过拟合gap.png")
