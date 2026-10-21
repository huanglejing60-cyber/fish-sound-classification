# -*- coding: utf-8 -*-
"""
=====================================================================
第 11 课：迁移学习——把别人练好的"耳朵"借来，只改最后一点点
---------------------------------------------------------------------
CNN14 有 ~8000 万个旋钮。它先在 AudioSet 的 200 万段声音(狗叫/汽车/音乐/
水流…527类)上把"耳朵"练好了。我们没有那么多鱼声，所以：
  · 不从头练它的耳朵（会过拟合、显存也不够），而是把大部分旋钮"冻结"锁住；
  · 把最后的"分类头"换成新的 4 类(鱼/应激/水/撞击)，只训练后面几层。
本脚本：
  第一部分：打印三种冻结策略分别开放多少旋钮（不训练，秒出）；
  第二部分：A 只训新头  vs  B 解冻最后2个卷积块微调，各跑12轮，看准确率
           和"真鱼召回"的差别（重现你脚本25→26的提升）。
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
LABELS=["fish","stress","water","impact"]; CN={"fish":"真鱼","stress":"麻醉应激","water":"水流设备","impact":"撞击"}
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); BATCH=4; ACCUM=2; EPOCHS=12
torch.manual_seed(42); np.random.seed(42)

# ---------------- CNN14 结构（和你脚本26完全一致）----------------
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
    def forward(self,x):   # 输入 (N,64,300)，模型内部补成图像
        x=x.unsqueeze(1).permute(0,2,3,1); x=self.bn0(x); x=x.permute(0,3,2,1)
        x=self.conv_block1(x); x=self.conv_block2(x); x=self.conv_block3(x)
        x=self.conv_block4(x); x=self.conv_block5(x); x=self.conv_block6(x)
        x=torch.mean(x,dim=3); x1,_=torch.max(x,dim=2); x=x1+torch.mean(x,dim=2)
        x=F.dropout(x,0.5,self.training); x=F.relu_(self.fc1(x)); emb=F.dropout(x,0.5,self.training)
        return self.fc_audioset(emb),emb

def 借耳朵():
    """加载527类预训练耳朵，再换成4类新头（新头随机，必须从头学）"""
    from safetensors.torch import load_file
    raw=load_file(PRETRAIN); sd={}
    for k,v in raw.items():
        if k.startswith("backbone."):
            nk=k[len("backbone."):]
            if not(nk.startswith("spectrogram_extractor") or nk.startswith("logmel_extractor")): sd[nk]=v
    m=Cnn14(527); m.load_state_dict(sd,strict=True)
    m.fc_audioset=nn.Linear(2048,4); init_layer(m.fc_audioset)
    return m

# ---------------- 数据 ----------------
def load_mel(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
print("读取165个音频...")
def 组装(rs):
    X=np.stack([load_mel(os.path.join(ROOT,r["类别"],r["文件名"])) for r in rs])
    Y=np.array([int(r["类别id"]) for r in rs]); return torch.from_numpy(X).float(),torch.from_numpy(Y).long()
Xtr,Ytr=组装(tr); Xva,Yva=组装(va)
mean,std=Xtr.mean(),Xtr.std()+1e-6; Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
Xtr,Ytr=Xtr.to(DEVICE),Ytr.to(DEVICE); Xva,Yva=Xva.to(DEVICE),Yva.to(DEVICE)
cnt=np.array([(Ytr.cpu()==i).sum().item() for i in range(4)]); w=len(tr)/(4*np.maximum(cnt,1))

# =============== 第一部分：三种冻结策略，各开放多少旋钮？ ===============
demo=借耳朵()
各层={n:sum(p.numel() for p in mod.parameters()) for n,mod in demo.named_modules() if isinstance(mod,(ConvBlock,nn.Linear,nn.BatchNorm2d))}
总=sum(p.numel() for p in demo.parameters())
def 参(块):
    s=sum(各层[f"conv_block{i}"] for i in 块)+各层["fc1"]+各层["fc_audioset"]
    return s
只头=参([]); 微调=参([5,6])
print("\n========== 每层旋钮数（从耳朵到分类头）==========")
for n in ["bn0","conv_block1","conv_block2","conv_block3","conv_block4","conv_block5","conv_block6","fc1","fc_audioset"]:
    print(f"  {n:14s} {各层[n]/1e6:7.2f} M")
print("="*52)
print(f"  策略A 只训新头(冻结全部卷积块)     开放 {只头/1e6:6.2f}M / {总/1e6:.0f}M = {只头/总:.1%}")
print(f"  策略B 解冻最后2块+fc1+头(脚本26)  开放 {微调/1e6:6.2f}M / {总/1e6:.0f}M = {微调/总:.1%}")
print(f"  策略C 全部解冻(从头训,不推荐)      开放 {总/1e6:6.1f}M / {总/1e6:.0f}M = 100%")
del demo

# =============== 第二部分：A 只训头 vs B 微调 ===============
def 训练(解冻块,标签):
    torch.manual_seed(42)
    m=借耳朵()
    for p in m.parameters(): p.requires_grad=False
    for i in 解冻块:
        for p in getattr(m,f"conv_block{i}").parameters(): p.requires_grad=True
    for p in m.fc1.parameters(): p.requires_grad=True
    for p in m.fc_audioset.parameters(): p.requires_grad=True
    冻=[f"conv_block{i}" for i in range(1,7) if i not in 解冻块]+["bn0"]
    m.to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEVICE))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    hist={"acc":[],"鱼召回":[]}
    def 置训练():
        m.train()
        for name,mod in m.named_modules():
            if isinstance(mod,nn.BatchNorm2d) and name.split(".")[0] in 冻: mod.eval()
    for ep in range(1,EPOCHS+1):
        置训练(); idx=torch.randperm(len(tr)); opt.zero_grad()
        for bi0 in range(0,len(tr),BATCH):
            bi=idx[bi0:bi0+BATCH]; xb,yb=Xtr[bi],Ytr[bi]
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
                out,_=m(xb); loss=crit(out,yb)/ACCUM
            scaler.scale(loss).backward()
            if (bi0//BATCH+1)%ACCUM==0 or (bi0+BATCH)>=len(tr):
                scaler.step(opt); scaler.update(); opt.zero_grad()
        m.eval()
        with torch.no_grad():
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
                pred=m(Xva)[0].argmax(1)
            acc=(pred==Yva).float().mean().item()
            鱼mask=Yva==0; 鱼rec=((pred==0)&鱼mask).sum().item()/鱼mask.sum().item()
        hist["acc"].append(acc); hist["鱼召回"].append(鱼rec)
        print(f"[{标签}] {ep:02d}轮 验证{acc:.0%} 真鱼召回{鱼rec:.0%}",flush=True)
    return hist

print("\n========== 策略A：只训新头（借来的耳朵完全不动）==========")
hA=训练([], "A只训头")
print("\n========== 策略B：解冻最后2块，让耳朵再熟悉鱼声 ==========")
hB=训练([5,6], "B微调")

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(hA["acc"],c="#9333ea",label="A 只训新头"); ax[0].plot(hB["acc"],c="#16a34a",label="B 解冻微调")
ax[0].set_title("四分类验证准确率"); ax[0].set_xlabel("轮次"); ax[0].set_ylim(0,1); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hA["鱼召回"],c="#9333ea",label="A 只训新头(耳朵对低频咕咕不敏感)")
ax[1].plot(hB["鱼召回"],c="#16a34a",label="B 解冻微调(能认出鱼)")
ax[1].set_title("关键：真鱼召回（别把鱼漏掉）"); ax[1].set_xlabel("轮次"); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第11课_迁移学习_只训头vs微调.png",dpi=140,bbox_inches="tight")
print(f"\n最终：A 只训头 验证{hA['acc'][-1]:.0%}/鱼召回{hA['鱼召回'][-1]:.0%}；"
      f"B 微调 验证{hB['acc'][-1]:.0%}/鱼召回{hB['鱼召回'][-1]:.0%}")
print("图在 学习/第11课_迁移学习_只训头vs微调.png")
