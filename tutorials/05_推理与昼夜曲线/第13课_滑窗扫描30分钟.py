# -*- coding: utf-8 -*-
"""
=====================================================================
第 13 课：让模型自动扫30分钟长录音——"滑窗"
---------------------------------------------------------------------
模型只会吃 3 秒片段，不会一次听30分钟。怎么办？把长录音像"切面包"一样
切成一连串 3 秒的小窗口，逐段送进去判断，这就叫【滑窗】。
本脚本：
  1. 读成鱼30分钟录音(96k→重采样32k)；
  2. 切成600个3秒窗口(第13课无重叠，步进=3秒)；
  3. 每段转mel→标准化→模型给4类概率，记录"鱼概率"；
  4. 存CSV(供第14课画曲线)，并画出"原始鱼概率随时间"图。
你会看到：哪几个3秒窗口模型认为有鱼。
=====================================================================
"""
import os, csv, time
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

WAV=r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
MODEL=r"E:\大黄鱼分析\models\cnn14_finetune.pt"
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
WIN=3.0; STEP=3.0; BATCH=16; THRESH=0.5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV=r"E:\大黄鱼分析\学习\第13课_30分钟扫描结果.csv"

# ---------- CNN14 结构（同训练脚本）----------
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

# ---------- 加载模型 ----------
ckpt=torch.load(MODEL,map_location=DEVICE,weights_only=False)
labels=ckpt["labels"]; mean=float(ckpt["mean"]); std=float(ckpt["std"]+1e-6)
模型=Cnn14(4); 模型.load_state_dict(ckpt["state_dict"]); 模型.to(DEVICE).eval()
print("模型加载完成，设备",DEVICE)

# ---------- 第1步：读长录音（96k→32k）----------
t0=time.time()
print("读取长录音并重采样到32k（约10-30秒）...")
y,_=librosa.load(WAV,sr=SR,mono=True)
总时长=len(y)/SR
print(f"总时长 {总时长:.0f} 秒 = {总时长/60:.1f} 分钟，读取耗时 {time.time()-t0:.0f}s")

# ---------- 第2步：切窗口（滑窗）----------
窗长=int(WIN*SR); 步进=int(STEP*SR)
起点=list(range(0,len(y)-窗长+1,步进))
print(f"窗口：每窗{WIN}s，步进{STEP}s，共 {len(起点)} 个窗口")
print("前3个窗口起始时间（秒）：",[round(s/SR,2) for s in 起点[:3]])

# ---------- 第3步：每段转mel ----------
t0=time.time()
M=np.zeros((len(起点),N_MELS,N_FRAMES),dtype=np.float32)
for i,s in enumerate(起点):
    seg=y[s:s+窗长]
    S=librosa.feature.melspectrogram(y=seg,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    if S.shape[1]>=N_FRAMES: S=S[:,:N_FRAMES]
    else: S=np.pad(S,((0,0),(0,N_FRAMES-S.shape[1])))
    M[i]=S
print(f"转mel完成，耗时{time.time()-t0:.0f}s，数组{M.shape}")

# ---------- 第4步：标准化 + 批量推理 ----------
X=torch.from_numpy((M-mean)/std).float().to(DEVICE)
概率=np.zeros((len(起点),4),dtype=np.float32)
t0=time.time()
with torch.no_grad():
    for s in range(0,len(起点),BATCH):
        out,_=模型(X[s:s+BATCH])
        概率[s:s+BATCH]=torch.softmax(out,dim=1).cpu().numpy()
print(f"推理完成，耗时{time.time()-t0:.0f}s")
鱼概率=概率[:,labels.index("fish")]
中点秒=np.array(起点)/SR+WIN/2
报鱼=鱼概率>=THRESH
print(f"\n鱼概率≥{THRESH} 的窗口数：{报鱼.sum()} / {len(起点)}")
if 报鱼.sum()>0:
    print(f"有鱼时间范围：{中点秒[报鱼].min()/60:.1f} ~ {中点秒[报鱼].max()/60:.1f} 分钟")

# ---------- 存CSV ----------
with open(OUT_CSV,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["窗口","起始秒","中点秒","分钟","鱼概率","应激概率","水概率","撞击概率"])
    for i in range(len(起点)):
        w.writerow([i,起点[i]/SR,中点秒[i],中点秒[i]/60]+[f"{v:.4f}" for v in 概率[i]])
print("已存CSV：",OUT_CSV)

# ---------- 画图：原始鱼概率随时间 ----------
分钟=中点秒/60
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(分钟,鱼概率,c="#2563eb",lw=1,alpha=.7,zorder=1)
ax.scatter(分钟[~报鱼],鱼概率[~报鱼],s=14,c="#94a3b8",label="鱼概率<0.5（不报）",zorder=2)
ax.scatter(分钟[报鱼],鱼概率[报鱼],s=46,c="#dc2626",label="鱼概率≥0.5（报鱼）",zorder=3)
ax.axhline(THRESH,ls="--",c="#dc2626",lw=1)
ax.set_xlabel("时间（分钟）"); ax.set_ylabel("模型给的鱼概率"); ax.set_ylim(0,1)
ax.set_title("第13课 模型滑窗扫描30分钟：每个3秒窗口的鱼概率（原始，未平滑）")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第13课_30分钟原始鱼概率.png",dpi=140,bbox_inches="tight")
print("图在 学习/第13课_30分钟原始鱼概率.png")
