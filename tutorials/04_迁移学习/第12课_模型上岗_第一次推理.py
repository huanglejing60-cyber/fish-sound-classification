# -*- coding: utf-8 -*-
"""
=====================================================================
第 12 课：模型上岗第一天——加载训好的CNN14，让它对新声音做判断
---------------------------------------------------------------------
训练时模型在"学习"（不断改旋钮）；推理时模型只"干活"（旋钮锁住，不再改）。
本脚本做三件事：
  1. 把训好的 cnn14_finetune.pt 加载回来（连同标准化用的mean/std）；
  2. 从验证集挑4段(每类1段，模型训练时没见过的)，打印它对每段的"4类信心"；
  3. 再让它考完全部40道验证题，给出准确率(和第7课呼应)。
核心概念：模型不会直接说"这是鱼"，它输出4个概率(加起来=100%)，
         哪个概率最大就判哪个。
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
CN={"fish":"真鱼","stress":"麻醉应激","water":"水流设备","impact":"撞击"}

# CNN14 结构（和训练脚本26一模一样，加载权重必须先有这个"空壳"）
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

# ===== 第1步：加载模型（空壳 + 灌权重 + 切到"上岗模式"）=====
print("="*60)
ckpt=torch.load(MODEL,map_location=DEVICE,weights_only=False)
print("模型文件里存了：",list(ckpt.keys()))
labels=ckpt["labels"]; mean=float(ckpt["mean"]); std=float(ckpt["std"]+1e-6)
print("类别顺序：",labels)
模型=Cnn14(4); 模型.load_state_dict(ckpt["state_dict"]); 模型.to(DEVICE)
模型.eval()    # 上岗模式：关掉dropout/BN更新，让输出稳定可复现
print("模型加载完成，参数旋钮数：",sum(p.numel() for p in 模型.parameters())/1e6,"M")

# 音频→mel（和训练时完全相同的配方，不能改）
def 读mel(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
def 预测(path):
    x=(读mel(path)-mean)/std                                  # 用训练时存的mean/std标准化
    x=torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)    # 补成 (1,64,300)
    with torch.no_grad():                                    # 推理：不算梯度，省显存、更快
        logits,_=模型(x)
        prob=torch.softmax(logits,dim=1)[0].cpu().numpy()    # 4个原始分→4个概率(加起来=1)
    return prob

# ===== 第2步：每类挑1段，打印4类信心 =====
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
va=[r for r in rows if r["split"]=="val"]
样例=[]
for lab in labels:
    r=next(r for r in va if r["类别"]==lab)
    样例.append(r)
print("="*60)
表头="真实类别 | "+" | ".join(f"{CN[l]:>5}" for l in labels)+"  => 模型判定"
print(表头)
print("-"*60)
矩阵=np.zeros((4,4))
for i,r in enumerate(样例):
    p=预测(os.path.join(ROOT,r["类别"],r["文件名"])); 矩阵[i]=p
    判=labels[int(p.argmax())]
    行=f"{CN[r['类别']]:>5} | "+" | ".join(f"{v:5.0%}" for v in p)+f"  => {CN[判]}"
    print(行)

# 画4x4概率热力图
fig,ax=plt.subplots(figsize=(7.5,6))
im=ax.imshow(矩阵,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels([CN[l] for l in labels],rotation=30,ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels([CN[l] for l in labels])
ax.set_xlabel("模型认为的概率"); ax.set_ylabel("真实声音")
for i in range(4):
    for j in range(4):
        ax.text(j,i,f"{矩阵[i,j]:.0%}",ha="center",va="center",color="white" if 矩阵[i,j]>0.5 else "black")
ax.set_title("第12课 模型上岗：每类听1段，它的4类信心\n(对角线上数字越大=判得越准越自信)")
plt.colorbar(im,fraction=0.046,label="概率")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第12课_模型上岗_4类信心.png",dpi=140,bbox_inches="tight")

# ===== 第3步：考完全部40道验证题 =====
对=0
for r in va:
    p=预测(os.path.join(ROOT,r["类别"],r["文件名"]))
    if labels[int(p.argmax())]==r["类别"]: 对+=1
print("="*60)
print(f"全部40个验证样本：对 {对} 个，准确率 {对/len(va):.0%}")
print("图在 学习/第12课_模型上岗_4类信心.png")
