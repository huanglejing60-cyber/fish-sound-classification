# -*- coding: utf-8 -*-
"""
=====================================================================
第 18 课 b：把数据增强加进训练，和"无增强"正面对比
---------------------------------------------------------------------
两组训练，除了"训练时是否增强"，其它完全相同（同种子、同超参、同模型）：
  基线：只用干净的162个样本
  增强：每轮对训练样本随机做 增益/时间平移/加真实环境噪声/SpecAugment
验证集始终不增强（公平考试）。
增强版模型存 cnn14_finetune_v3_aug.pt（不覆盖旧模型）。
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
PRETRAIN=os.path.join(MODEL_DIR,"cnn14_hf.safetensors"); SAVE=os.path.join(MODEL_DIR,"cnn14_finetune_v3_aug.pt")
LABELS=["fish","stress","water","impact"]
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); BATCH=4; ACCUM=2; EPOCHS=20

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
print(f"train {len(tr)} val {len(va)}，读波形…",flush=True)
波tr=[读波形(os.path.join(ROOT,r["类别"],r["文件名"])) for r in tr]; Ytr=np.array([int(r["类别id"]) for r in tr])
Mva=np.stack([转mel(读波形(os.path.join(ROOT,r["类别"],r["文件名"]))) for r in va]); Yva=np.array([int(r["类别id"]) for r in va])
# 噪声池：用真实的水流/设备/应激声当背景噪声（不用鱼，避免把别的鱼声当噪声）
噪声池=[读波形(os.path.join(ROOT,r["类别"],r["文件名"])) for r in tr if r["类别"] in ("water","impact")]
# 干净mel的mean/std（固定，增强也用它）
Mtr0=np.stack([转mel(y) for y in 波tr]); mean,std=Mtr0.mean(),Mtr0.std()+1e-6
Xtr_clean=torch.from_numpy((Mtr0-mean)/std).float().to(DEVICE); Ytr_t=torch.from_numpy(Ytr).long().to(DEVICE)
Xva=torch.from_numpy((Mva-mean)/std).float().to(DEVICE); Yva_t=torch.from_numpy(Yva).long().to(DEVICE)
cnt=np.array([(Ytr==i).sum() for i in range(4)]); w=len(tr)/(4*np.maximum(cnt,1))
rng=np.random.default_rng(0)

# ---------- 增强 ----------
def 波形增强(y):
    y=y.copy()*rng.uniform(0.6,1.2)                       # 随机音量
    sh=int(rng.integers(-int(0.4*SR),int(0.4*SR))); y=np.roll(y,sh)  # 时间平移
    if rng.random()<0.5:                                # 50%叠真实环境噪声
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
    hist={"acc":[],"鱼精度":[],"鱼召回":[]}; best=-1; bestsd=None
    for ep in range(1,EPOCHS+1):
        m.train()
        for name,mod in m.named_modules():
            if isinstance(mod,nn.BatchNorm2d) and name.split(".")[0] in 冻: mod.eval()
        Xt=增强epoch() if 用增强 else Xtr_clean
        idx=torch.randperm(len(tr)); opt.zero_grad()
        for bi0 in range(0,len(tr),BATCH):
            bi=idx[bi0:bi0+BATCH]
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
                out,_=m(Xt[bi]); loss=crit(out,Ytr_t[bi])/ACCUM
            scaler.scale(loss).backward()
            if (bi0//BATCH+1)%ACCUM==0 or (bi0+BATCH)>=len(tr):
                scaler.step(opt); scaler.update(); opt.zero_grad()
        m.eval()
        with torch.no_grad():
            pr=[]
            for i in range(0,len(va),BATCH):
                with torch.autocast("cuda",enabled=DEVICE.type=="cuda"): lo,_=m(Xva[i:i+BATCH])
                pr.append(torch.softmax(lo,1).float())
            pred=torch.cat(pr).argmax(1); acc=(pred==Yva_t).float().mean().item()
            鱼m=Yva_t==0; TP=((pred==0)&鱼m).sum().item(); FP=((pred==0)&~鱼m).sum().item(); FN=((pred!=0)&鱼m).sum().item()
            prec=TP/max(TP+FP,1); rec=TP/max(TP+FN,1)
        hist["acc"].append(acc); hist["鱼精度"].append(prec); hist["鱼召回"].append(rec)
        if acc>best: best=acc; bestsd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        print(f"[{标签}] {ep:02d}轮 验证{acc:.0%} 鱼精度{prec:.0%} 鱼召回{rec:.0%}",flush=True)
    return hist,bestsd,best

print("\n===== 基线：无增强 =====",flush=True)
h0,sd0,b0=训练(False,"无增强")
SAVE_BASE=os.path.join(MODEL_DIR,"cnn14_finetune_v3_base_noaug.pt")
torch.save({"state_dict":sd0,"mean":float(mean),"std":float(std),"labels":LABELS},SAVE_BASE)
print("\n===== 增强：每轮随机变形 =====",flush=True)
h1,sd1,b1=训练(True,"增强")
torch.save({"state_dict":sd1,"mean":float(mean),"std":float(std),"labels":LABELS},SAVE)
print(f"\n两个模型已存：{os.path.basename(SAVE_BASE)} / {os.path.basename(SAVE)}")

print(f"最佳验证准确率：无增强 {b0:.0%} → 增强 {b1:.0%}")

# ---------- 鲁棒性测试：把验证集变"难"，看两个模型谁还认得出鱼 ----------
# 模拟16T首次发声会遇到的真实困难：弱声(离得远/刚开始叫)、环境嘈杂、中间漏听
def 加载(sd):
    mm=借耳朵(); mm.load_state_dict(sd); mm.to(DEVICE); mm.eval(); return mm
m0=加载(sd0); m1=加载(sd1)
波va=[读波形(os.path.join(ROOT,r["类别"],r["文件名"])) for r in va]
def 标(M): return torch.from_numpy((M-mean)/std).float().to(DEVICE)
Xc={}
Xc["干净"]=Xva
Xc["弱声×0.3"]=标(np.stack([转mel(y*0.3) for y in 波va]))
Xc["嘈杂"]=标(np.stack([转mel(y+0.5*噪声池[i%len(噪声池)]) for i,y in enumerate(波va)]))
M漏=Mva.copy(); rt=np.random.default_rng(7)
for i in range(len(va)):
    t0=int(rt.integers(30,250)); M漏[i,:,t0:t0+50]=M漏[i].min()
Xc["漏帧"]=标(M漏)
def 评估(mm,X):
    pp=[]
    with torch.no_grad():
        for i in range(0,len(va),BATCH):
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"): lo,_=mm(X[i:i+BATCH])
            pp.append(torch.softmax(lo,1).float().argmax(1))
    p=torch.cat(pp); 鱼m=Yva_t==0
    TP=((p==0)&鱼m).sum().item(); FP=((p==0)&~鱼m).sum().item(); FN=((p!=0)&鱼m).sum().item()
    return (p==Yva_t).float().mean().item(), TP/max(TP+FP,1), TP/max(TP+FN,1)
print("\n===== 鲁棒性测试（验证集变难，两个模型公平PK）=====")
print("条件        | 无增强 精度/召回 | 增强 精度/召回")
res={}
for 条件,X in Xc.items():
    a0,p0,r0=评估(m0,X); a1,p1,r1=评估(m1,X); res[条件]=((p0,r0),(p1,r1))
    print(f"{条件:9s} |   {p0:3.0%} / {r0:3.0%}    |  {p1:3.0%} / {r1:3.0%}")

# 图1：训练曲线
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
for k,(c,yl) in enumerate([("acc","验证准确率"),("鱼精度","真鱼精度"),("鱼召回","真鱼召回")]):
    ax[k].plot(h0[c],c="#9ca3af",label="无增强"); ax[k].plot(h1[c],c="#2563eb",label="数据增强")
    ax[k].set_title(yl); ax[k].set_ylim(0,1); ax[k].set_xlabel("轮次"); ax[k].legend(); ax[k].grid(alpha=.3)
plt.suptitle("第18课b 训练曲线：无增强 vs 增强（干净小验证集上，增强不一定提分）")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第18课b1_训练曲线.png",dpi=140,bbox_inches="tight")
# 图2：困难条件下的真鱼召回（核心）
conds=list(Xc.keys()); x=np.arange(len(conds)); wd=0.35
r0=[res[c][0][1] for c in conds]; r1=[res[c][1][1] for c in conds]
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-wd/2,r0,wd,label="无增强",color="#9ca3af"); ax.bar(x+wd/2,r1,wd,label="数据增强",color="#2563eb")
for xi,v in zip(x-wd/2,r0): ax.text(xi,v+0.02,f"{v:.0%}",ha="center",fontsize=10)
for xi,v in zip(x+wd/2,r1): ax.text(xi,v+0.02,f"{v:.0%}",ha="center",fontsize=10,color="#1d4ed8")
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0,1.08)
ax.set_ylabel("真鱼召回（困难条件下还认不认得鱼）"); ax.set_title("第18课b 数据增强的真正价值：弱声/嘈杂/漏帧时更稳")
ax.legend(); ax.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第18课b2_鲁棒性测试.png",dpi=140,bbox_inches="tight")
print("图在 学习/第18课b1_训练曲线.png 和 第18课b2_鲁棒性测试.png")
