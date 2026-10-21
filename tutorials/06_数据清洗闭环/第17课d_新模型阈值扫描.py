# -*- coding: utf-8 -*-
"""第17课d：用清洗后新模型在验证集上扫描置信度阈值，找'零误报'工作点（导师：宁漏勿错）。"""
import os, csv
import numpy as np, librosa
import torch, torch.nn as nn, torch.nn.functional as F
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL=r"E:\大黄鱼分析\models\cnn14_finetune_v2_clean.pt"
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
def 读mel(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    return np.pad(librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32),((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
ckpt=torch.load(MODEL,map_location=DEVICE,weights_only=False)
labels=ckpt["labels"]; mean=float(ckpt["mean"]); std=float(ckpt["std"]+1e-6)
m=Cnn14(4); m.load_state_dict(ckpt["state_dict"]); m.to(DEVICE); m.eval()
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
va=[r for r in rows if r["split"]=="val"]
probs=[]; ys=[]
for r in va:
    x=(读mel(os.path.join(ROOT,r["类别"],r["文件名"]))-mean)/std
    x=torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p=torch.softmax(m(x)[0],1)[0,0].item()
    probs.append(p); ys.append(int(r["类别id"])==0)
probs=np.array(probs); ys=np.array(ys)
nf=ys.sum(); print(f"验证集{len(va)}个：真鱼{nf}个，非鱼{len(va)-nf}个")
print("门槛T | 报鱼数 | 真抓到 误报 漏掉 | 精度   召回")
print("-"*58)
推荐=None
for T in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]:
    pred=probs>=T; TP=((pred)&ys).sum(); FP=((pred)&~ys).sum(); FN=((~pred)&ys).sum()
    prec=TP/max(TP+FP,1); rec=TP/max(TP+FN,1)
    if FP==0 and 推荐 is None: 推荐=T
    print(f"{T:.2f} | {pred.sum():5d} | {TP:5d} {FP:4d} {FP:5d} {FN:4d} | {prec:4.0%} {rec:5.0%}")
print("-"*58)
if 推荐 is not None:
    rec_=(probs>=推荐)&ys
    print(f"推荐零误报工作点 T={推荐}：精度100%，召回{rec_.sum()/nf:.0%}")
