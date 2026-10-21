# -*- coding: utf-8 -*-
"""
=====================================================================
第15课a：批量扫描一整天48个文件，画24小时昼夜发声曲线
---------------------------------------------------------------------
把第13/14课的生产线自动套到每个30分钟文件上：
  每个文件 → 滑窗 → 鱼概率 → 1分钟平滑 → 统计"成片发声强度"
然后按时间排开，画出大黄鱼一天里什么时候叫。
批量处理要点：自动遍历文件、逐个处理(用完释放内存)、单个出错不中断、存进度。
=====================================================================
"""
import os, re, csv, time, glob, datetime as dt
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

DIR=r"E:\大黄鱼数据\#30054#[020]@20250313_085156"
MODEL=r"E:\大黄鱼分析\models\cnn14_finetune.pt"
OUT_CSV=r"E:\大黄鱼分析\学习\第15课_全天扫描结果.csv"
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
WIN=3.0; BATCH=16; FISH_T=0.5; SEG_T=0.30
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CNN14 ----------
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
labels=ckpt["labels"]; fi=labels.index("fish"); mean=float(ckpt["mean"]); std=float(ckpt["std"]+1e-6)
模型=Cnn14(4); 模型.load_state_dict(ckpt["state_dict"]); 模型.to(DEVICE).eval()
窗长=int(WIN*SR)

def 扫一个文件(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    起点=list(range(0,len(y)-窗长+1,窗长))
    M=np.zeros((len(起点),N_MELS,N_FRAMES),dtype=np.float32)
    for i,s in enumerate(起点):
        seg=y[s:s+窗长]
        S=librosa.feature.melspectrogram(y=seg,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
        S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
        M[i]=(S[:,:N_FRAMES] if S.shape[1]>=N_FRAMES else np.pad(S,((0,0),(0,N_FRAMES-S.shape[1]))))
    X=torch.from_numpy((M-mean)/std).float().to(DEVICE)
    p=np.zeros(len(起点),dtype=np.float32)
    with torch.no_grad():
        for s in range(0,len(起点),BATCH):
            out,_=模型(X[s:s+BATCH]); p[s:s+BATCH]=torch.softmax(out,dim=1)[:,fi].cpu().numpy()
    sm=np.convolve(p,np.ones(20)/20,mode="same")
    return int((p>=FISH_T).sum()), int((sm>=SEG_T).sum()), float(sm.max())  # 报鱼窗口,成片窗口,峰值

files=sorted(os.path.join(DIR,f) for f in os.listdir(DIR) if f.lower().endswith(".wav"))
print(f"共{len(files)}个文件，开始批量扫描…",flush=True)
记录=[]; t0=time.time()
base=None
for n,fp in enumerate(files):
    fname=os.path.basename(fp)
    m=re.search(r"@(\d{8})_(\d{6})@MID",fname)
    t_start=dt.datetime.strptime(m.group(1)+m.group(2),"%Y%m%d%H%M%S")
    if base is None: base=t_start.replace(hour=0,minute=0,second=0)
    rel=(t_start-base).total_seconds()/3600
    try:
        nfish,nseg,peak=扫一个文件(fp)
    except Exception as e:
        print(f"[{n+1}/{len(files)}] {fname} 出错跳过:{e}",flush=True); continue
    记录.append((fname,t_start.strftime("%m-%d %H:%M"),rel,nfish,nseg,peak))
    if (n+1)%4==0 or n==len(files)-1:
        print(f"[{n+1}/{len(files)}] {t_start.strftime('%H:%M')} 成片{nseg}窗 峰值{peak:.2f} 累计{time.time()-t0:.0f}s",flush=True)

with open(OUT_CSV,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["文件名","起始时刻","相对小时","报鱼窗口数","成片窗口数","平滑峰值"])
    for r in 记录: w.writerow(r)
print(f"全部完成，耗时{time.time()-t0:.0f}s，已存{OUT_CSV}",flush=True)

# ---------- 画24小时曲线 ----------
rel=np.array([r[2] for r in 记录]); seg=np.array([r[4] for r in 记录]); peak=np.array([r[5] for r in 记录])
钟点=(rel)%24
fig,ax=plt.subplots(figsize=(14,5.5))
# 夜间阴影(18:00-次日06:00)
for lo,hi in [(18,24),(24,30)]:
    ax.axvspan(lo,hi,color="#1e3a8a",alpha=.07)
colors=["#1e3a8a" if (h>=18 or h<6) else "#93c5fd" for h in rel]
ax.bar(rel,seg*3/60,width=0.45,color=colors,zorder=2)  # 成片窗口数×3秒→分钟
peak_norm = peak*(seg.max()*3/60)/peak.max() if peak.max()>0 else np.zeros_like(peak)
ax.plot(rel,peak_norm,color="#dc2626",lw=1.5,marker="o",ms=3,label="平滑峰值(归一化)")
ax.set_xlabel("钟点"); ax.set_ylabel("每30分钟成片发声时长（分钟）")
xt=np.arange(8,33,2); ax.set_xticks(xt); ax.set_xticklabels([f"{int(h%24):02d}:00" for h in xt])
ax.set_xlim(7.5,33)
pk=int(np.argmax(seg))
ax.annotate(f"产卵高峰\n{记录[pk][1]}",(rel[pk],seg[pk]*3/60),xytext=(rel[pk]-3,seg[pk]*3/60*0.85),
            arrowprops=dict(arrowstyle="->",color="#dc2626"),color="#dc2626",ha="center",fontsize=10)
ax.set_title("第15课 大黄鱼24小时昼夜发声曲线（CNN14批量扫描，蓝柱=成片发声，深蓝=夜间）")
ax.legend(); ax.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第15课_24小时昼夜曲线.png",dpi=140,bbox_inches="tight")
print("图在 学习/第15课_24小时昼夜曲线.png",flush=True)
