# -*- coding: utf-8 -*-
"""
=====================================================================
第 17 课：用"清洗后"的干净标签重训 CNN14（策略B：解冻最后2块微调）
---------------------------------------------------------------------
上一版模型 cnn14_finetune.pt 是用 165 个、含20个错标鱼的数据集训的。
现在标签洗干净（162个，fish 32 / stress 37 / water 48 / impact 45），
同样的方法再训一次，另存为 cnn14_finetune_v2_clean.pt（不覆盖旧模型）。
训练后在验证集上报告：准确率、真鱼的精度/召回、混淆矩阵。
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
SAVE=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt")
LABELS=["fish","stress","water","impact"]; CN={"fish":"真鱼","stress":"应激","water":"水流设备","impact":"撞击"}
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); BATCH=4; ACCUM=2; EPOCHS=20
torch.manual_seed(42); np.random.seed(42)

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
    m.fc_audioset=nn.Linear(2048,4); init_layer(m.fc_audioset)
    return m

def load_mel(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
print(f"清洗后：train {len(tr)}  val {len(va)}，读mel...")
def 组装(rs):
    X=np.stack([load_mel(os.path.join(ROOT,r["类别"],r["文件名"])) for r in rs])
    Y=np.array([int(r["类别id"]) for r in rs]); return torch.from_numpy(X).float(),torch.from_numpy(Y).long()
Xtr,Ytr=组装(tr); Xva,Yva=组装(va)
mean,std=Xtr.mean(),Xtr.std()+1e-6; Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
Xtr,Ytr=Xtr.to(DEVICE),Ytr.to(DEVICE); Xva,Yva=Xva.to(DEVICE),Yva.to(DEVICE)
cnt=np.array([(Ytr.cpu()==i).sum().item() for i in range(4)]); w=len(tr)/(4*np.maximum(cnt,1))
print("训练集类别数 fish/stress/water/impact =",cnt.tolist())

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
hist={"acc":[],"鱼精度":[],"鱼召回":[],"loss":[]}; best_acc=-1; best_sd=None
def 置训练():
    m.train()
    for name,mod in m.named_modules():
        if isinstance(mod,nn.BatchNorm2d) and name.split(".")[0] in 冻: mod.eval()
for ep in range(1,EPOCHS+1):
    置训练(); idx=torch.randperm(len(tr)); opt.zero_grad(); tot=0
    for bi0 in range(0,len(tr),BATCH):
        bi=idx[bi0:bi0+BATCH]; xb,yb=Xtr[bi],Ytr[bi]
        with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
            out,_=m(xb); loss=crit(out,yb)/ACCUM
        scaler.scale(loss).backward(); tot+=loss.item()*ACCUM
        if (bi0//BATCH+1)%ACCUM==0 or (bi0+BATCH)>=len(tr):
            scaler.step(opt); scaler.update(); opt.zero_grad()
    m.eval()
    with torch.no_grad():
        pr=[]
        for i in range(0,len(va),BATCH):
            with torch.autocast("cuda",enabled=DEVICE.type=="cuda"):
                lo,_=m(Xva[i:i+BATCH])
            pr.append(torch.softmax(lo,1).float())
        P=torch.cat(pr); pred=P.argmax(1)
        acc=(pred==Yva).float().mean().item()
        鱼m=Yva==0
        TP=((pred==0)&鱼m).sum().item(); FP=((pred==0)&~鱼m).sum().item(); FN=((pred!=0)&鱼m).sum().item()
        prec=TP/max(TP+FP,1); rec=TP/max(TP+FN,1)
    hist["acc"].append(acc); hist["鱼精度"].append(prec); hist["鱼召回"].append(rec); hist["loss"].append(tot/len(tr)*BATCH)
    if acc>best_acc: best_acc=acc; best_sd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    print(f"{ep:02d}轮 loss{hist['loss'][-1]:.3f} 验证{acc:.0%} 鱼精度{prec:.0%} 鱼召回{rec:.0%}",flush=True)

# 保存最佳模型
torch.save({"state_dict":best_sd,"mean":float(mean),"std":float(std),"labels":LABELS},SAVE)
print("\n已保存新模型(不覆盖旧的)：",os.path.basename(SAVE))

# 用最佳模型在val上出混淆矩阵
m.load_state_dict(best_sd); m.eval()
with torch.no_grad():
    pr=[]
    for i in range(0,len(va),BATCH):
        with torch.autocast("cuda",enabled=DEVICE.type=="cuda"): lo,_=m(Xva[i:i+BATCH])
        pr.append(torch.softmax(lo,1).float())
    P=torch.cat(pr); pred=P.argmax(1)
cm=np.zeros((4,4),dtype=int)
for t,p in zip(Yva.cpu().numpy(),pred.cpu().numpy()): cm[t,p]+=1
print("\n【混淆矩阵】行=真实，列=预测")
print("        "+" ".join(f"{CN[l]:>5}" for l in LABELS))
for i,l in enumerate(LABELS):
    print(f"{CN[l]:>5}  "+" ".join(f"{cm[i,j]:>5}" for j in range(4)))
print(f"\n最佳验证准确率 {best_acc:.0%}")
print(f"真鱼：精度={cm[0,0]/max(cm[:,0].sum(),1):.0%}（报鱼里真鱼） 召回={cm[0,0]/max(cm[0].sum(),1):.0%}（{cm[0].sum()}个真鱼抓到几个）")

fig,ax=plt.subplots(1,2,figsize=(13,4.6))
ax[0].plot(hist["acc"],c="#2563eb",label="验证准确率"); ax[0].plot(hist["鱼精度"],c="#dc2626",label="真鱼精度"); ax[0].plot(hist["鱼召回"],c="#16a34a",label="真鱼召回")
ax[0].set_title("清洗后重训 学习曲线"); ax[0].set_xlabel("轮次"); ax[0].set_ylim(0,1); ax[0].legend(); ax[0].grid(alpha=.3)
im=ax[1].imshow(cm,cmap="Blues"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels([CN[l] for l in LABELS],rotation=30,ha="right")
ax[1].set_yticks(range(4)); ax[1].set_yticklabels([CN[l] for l in LABELS]); ax[1].set_xlabel("预测"); ax[1].set_ylabel("真实")
for i in range(4):
    for j in range(4): ax[1].text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
ax[1].set_title("验证集混淆矩阵")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第17课_清洗后重训结果.png",dpi=140,bbox_inches="tight")
print("图在 学习/第17课_清洗后重训结果.png")
