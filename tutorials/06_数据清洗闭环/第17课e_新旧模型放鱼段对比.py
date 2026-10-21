# -*- coding: utf-8 -*-
"""
第17课e：数据闭环的"期末考试"——用清洗前后两个模型扫同样的关键时段
  早晨放鱼干扰段 08:51/09:21/09:51（旧模型曾大面积误报）
  喂食段 16:21
  傍晚真产卵段 19:21/19:51
期望：新模型在早晨不再误报(下降)，同时傍晚真产卵仍能检出(保持)。
"""
import os, re
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
DIR=r"E:\大黄鱼数据\#30054#[020]@20250313_085156"
MODELS={"旧模型(脏标签)":r"E:\大黄鱼分析\models\cnn14_finetune.pt",
        "新模型(清洗后)":r"E:\大黄鱼分析\models\cnn14_finetune_v2_clean.pt"}
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
WIN=3.0; BATCH=16; FISH_T=0.5; SEG_T=0.30
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
def 加载(path):
    ck=torch.load(path,map_location=DEVICE,weights_only=False)
    lab=ck["labels"]; fi=lab.index("fish"); mean=float(ck["mean"]); std=float(ck["std"]+1e-6)
    m=Cnn14(4); m.load_state_dict(ck["state_dict"]); m.to(DEVICE).eval()
    return m,fi,mean,std
M={}; 
for name,mp in MODELS.items(): M[name]=加载(mp)
print("两个模型加载完成",flush=True)
窗长=int(WIN*SR)
# 关键时段
时段=[("08:51 放鱼(干扰)","085156"),("09:21 放鱼(干扰)","092156"),("09:51 放鱼(干扰)","095156"),
      ("16:21 喂食","162156"),("19:21 产卵(真鱼)","192156"),("19:51 产卵(真鱼)","195156")]
def 找文件(hm):
    for f in os.listdir(DIR):
        if f.lower().endswith(".wav") and f"_{hm}@MID" in f: return os.path.join(DIR,f)
    return None
def mel数组(y):
    起点=list(range(0,len(y)-窗长+1,窗长)); A=np.zeros((len(起点),N_MELS,N_FRAMES),dtype=np.float32)
    for i,s in enumerate(起点):
        seg=y[s:s+窗长]
        S=librosa.feature.melspectrogram(y=seg,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
        S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
        A[i]=(S[:,:N_FRAMES] if S.shape[1]>=N_FRAMES else np.pad(S,((0,0),(0,N_FRAMES-S.shape[1]))))
    return A
res={}  # (时段,模型)->(p,sm,nfish,nseg,peak)
for label,hm in 时段:
    fp=找文件(hm); print("扫描",label,flush=True)
    y,_=librosa.load(fp,sr=SR,mono=True); A0=mel数组(y)
    for name in MODELS:
        m,fi,mean,std=M[name]; X=torch.from_numpy((A0-mean)/std).float().to(DEVICE)
        p=np.zeros(len(A0),dtype=np.float32)
        with torch.no_grad():
            for s in range(0,len(A0),BATCH):
                o,_=m(X[s:s+BATCH]); p[s:s+BATCH]=torch.softmax(o,1)[:,fi].cpu().numpy()
        sm=np.convolve(p,np.ones(20)/20,mode="same")
        res[(label,name)]=(p,sm,int((p>=FISH_T).sum()),int((sm>=SEG_T).sum()),float(sm.max()))
    del y,A0
print("\n========== 新旧模型对比（报鱼窗口 / 成片窗口 / 峰值）==========")
print(f"{'时段':<18}{'旧模型成片':>10}{'新模型成片':>10}{'旧报鱼':>8}{'新报鱼':>8}")
for label,hm in 时段:
    o=res[(label,"旧模型(脏标签)")]; n=res[(label,"新模型(清洗后)")]
    print(f"{label:<18}{o[3]:>10}{n[3]:>10}{o[2]:>8}{n[2]:>8}")
# 画图
fig,ax=plt.subplots(2,2,figsize=(14,8),sharex=True)
for col,(label,hm) in enumerate([("09:21 放鱼(干扰)","092156"),("19:21 产卵(真鱼)","192156")]):
    for row,name in enumerate(["旧模型(脏标签)","新模型(清洗后)"]):
        a=ax[row,col]; p,sm,nf,ns,pk=res[(label,name)]
        x=np.arange(len(p))*3/60
        a.plot(x,p,color="#93c5fd",lw=.7,label="逐窗鱼概率")
        a.plot(x,sm,color="#dc2626",lw=1.6,label="1分钟平滑")
        a.axhline(FISH_T,color="#6b7280",ls="--",lw=.8); a.axhline(SEG_T,color="#f59e0b",ls=":",lw=.8)
        a.set_ylim(0,1); a.set_ylabel("鱼概率"); a.grid(alpha=.3)
        a.set_title(f"{name} · {label}（成片{ns}窗）",fontsize=11)
        a.legend(loc="upper right",fontsize=8)
for a in ax[1]: a.set_xlabel("分钟")
plt.suptitle("第17课 数据闭环验收：新模型应'放鱼段降下来、产卵段依然高'",fontsize=13)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第17课_新旧模型放鱼段验收.png",dpi=140,bbox_inches="tight")
print("图在 学习/第17课_新旧模型放鱼段验收.png")

# 柱状对比：6个时段新旧成片窗口数
fig2,ax2=plt.subplots(figsize=(13,5))
labs=[t[0] for t in 时段]; x=np.arange(len(labs)); wd=.38
oldv=[res[(t,"旧模型(脏标签)")][3] for t in labs]; newv=[res[(t,"新模型(清洗后)")][3] for t in labs]
b1=ax2.bar(x-wd/2,oldv,wd,label="旧模型(脏标签)",color="#fca5a5")
b2=ax2.bar(x+wd/2,newv,wd,label="新模型(清洗后)",color="#60a5fa")
ax2.bar_label(b1,fontsize=9); ax2.bar_label(b2,fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(labs,rotation=20,ha="right",fontsize=9)
ax2.set_ylabel("成片鱼声窗口数（平滑≥0.30）"); ax2.legend(); ax2.grid(alpha=.3,axis="y")
ax2.axvspan(-.5,2.5,color="#f59e0b",alpha=.08)
ax2.text(1,max(oldv)*.95,"早晨放鱼干扰区\n（新模型应降到接近0）",ha="center",color="#b45309",fontsize=10)
ax2.set_title("第17课 数据闭环验收：新模型误报大幅下降，同时真产卵依然检出")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第17课_新旧模型成片窗口对比.png",dpi=140,bbox_inches="tight")
print("图在 学习/第17课_新旧模型成片窗口对比.png")
