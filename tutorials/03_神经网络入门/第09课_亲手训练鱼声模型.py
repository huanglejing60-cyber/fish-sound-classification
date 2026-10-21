# -*- coding: utf-8 -*-
"""
=====================================================================
第 09 课：用你真实的鱼声语谱图，亲手训练一个神经网络
---------------------------------------------------------------------
它就是第8课玩具训练的"放大版"，5步循环完全一样：
  数据：240个二维点  ->  125张真实mel语谱图(64×300)
  网络：80个旋钮     ->  小CNN约30万个旋钮
  输出：分2类        ->  分4类(鱼/应激/水流/撞击)
每轮在125个训练样本上"学"，再到40个没见过的验证样本上"考"。
你会亲眼看到：训练loss下降、验证准确率上升，最后训练出自己的模型。
=====================================================================
"""
import os, csv
import numpy as np, librosa
import torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
torch.manual_seed(42); np.random.seed(42)

ROOT=r"E:\大黄鱼分析\dataset_train_v1"; DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS=["fish","stress","water","impact"]; CN={"fish":"真鱼","stress":"麻醉应激","water":"水流设备","impact":"撞击"}
SR=32000; N_MELS=64; N_FRAMES=300
EPOCHS=50

# ---------- 1. 读数据：每个wav -> 一张64×300的mel语谱图（就是模型的"题目图片"）----------
def 读mel(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=N_MELS,fmin=50,fmax=14000)
    S=librosa.power_to_db(S).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

rows=[]
with open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print("正在读取并转换165个音频为mel语谱图（约1分钟）...")
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
def 组装(rs):
    X=np.stack([读mel(os.path.join(ROOT,r["类别"],r["文件名"])) for r in rs])[:,None,:,:]  # (N,1,64,300)
    Y=np.array([int(r["类别id"]) for r in rs])
    return torch.from_numpy(X).float(), torch.from_numpy(Y).long()
Xtr,Ytr=组装(tr); Xva,Yva=组装(va)
# 标准化：用训练集的均值/标准差把图片缩放（只算训练集，防止偷看验证集）
mean,std=Xtr.mean(),Xtr.std()+1e-6; Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
Xtr,Ytr=Xtr.to(DEVICE),Ytr.to(DEVICE); Xva,Yva=Xva.to(DEVICE),Yva.to(DEVICE)
print(f"训练{len(tr)}张 / 验证{len(va)}张；设备{DEVICE}")

# ---------- 2. 小CNN：把mel图当成小图片，用卷积层看局部纹理，最后输出4类分数----------
class 卷积块(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),
                               nn.ReLU(),nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)
class 小CNN(nn.Module):
    def __init__(self,n=4):
        super().__init__()
        self.特征=nn.Sequential(卷积块(1,16),卷积块(16,32),卷积块(32,64),卷积块(64,128))
        self.池化=nn.AdaptiveAvgPool2d((1,1)); self.丢弃=nn.Dropout(0.3); self.分类=nn.Linear(128,n)
    def forward(self,x):
        x=self.特征(x); x=self.池化(x).flatten(1); x=self.丢弃(x); return self.分类(x)
模型=小CNN(4).to(DEVICE)
print(f"小CNN旋钮(参数)总数：{sum(p.numel() for p in 模型.parameters())/1e4:.1f}万")

# ---------- 3. 损失函数 + 优化器（学习率1e-3，小步稳走）----------
损失=nn.CrossEntropyLoss()
优化器=torch.optim.Adam(模型.parameters(),lr=1e-3,weight_decay=1e-4)

# ---------- 4. 训练循环：就是第8课那5步，重复EPOCHS轮----------
历史={"tr_loss":[],"va_loss":[],"tr_acc":[],"va_acc":[]}; 最佳acc=0; 最佳权重=None
for ep in range(1,EPOCHS+1):
    # ===== 训练（在125张训练图上）=====
    模型.train()
    优化器.zero_grad()          # ① 清零
    out=模型(Xtr)               # ② 前向：给每张图判4类
    l=损失(out,Ytr)            # ③ 对答案算loss
    l.backward()               # ④ 反向：算出每个旋钮怎么改
    优化器.step()              # ⑤ 更新旋钮
    tr_acc=(out.argmax(1)==Ytr).float().mean().item()
    # ===== 验证（在40张没见过的图上，只考不学，所以no_grad）=====
    模型.eval()
    with torch.no_grad():
        vo=模型(Xva); vl=损失(vo,Yva)
        va_acc=(vo.argmax(1)==Yva).float().mean().item()
    历史["tr_loss"].append(l.item()); 历史["va_loss"].append(vl.item())
    历史["tr_acc"].append(tr_acc); 历史["va_acc"].append(va_acc)
    if va_acc>最佳acc:
        最佳acc=va_acc; 最佳权重={k:v.cpu().clone() for k,v in 模型.state_dict().items()}
    if ep%5==0 or ep==1:
        print(f"第{ep:02d}轮  训练loss {l.item():.3f} 训练acc {tr_acc:.0%} | 验证loss {vl.item():.3f} 验证acc {va_acc:.0%}（最佳{最佳acc:.0%}）")

# ---------- 5. 保存模型 + 画训练曲线和混淆矩阵----------
os.makedirs(r"E:\大黄鱼分析\models",exist_ok=True)
torch.save({"state_dict":最佳权重,"mean":mean.item(),"std":std.item(),"labels":LABELS},
           r"E:\大黄鱼分析\models\教学_小CNN_第9课.pt")
模型.load_state_dict(最佳权重); 模型.eval()
with torch.no_grad():
    预测=模型(Xva).argmax(1).cpu().numpy(); 真实=Yva.cpu().numpy()
cm=confusion_matrix(真实,预测,labels=list(range(4)))
cmn=cm/np.maximum(cm.sum(1,keepdims=True),1)

fig,ax=plt.subplots(1,2,figsize=(12,4.3))
ax[0].plot(历史["tr_loss"],label="训练loss"); ax[0].plot(历史["va_loss"],label="验证loss")
ax[0].set_title("loss：随训练下降（训练在学，验证检验是否真懂）"); ax[0].set_xlabel("轮次"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(历史["tr_acc"],label="训练准确率"); ax[1].plot(历史["va_acc"],label="验证准确率")
ax[1].set_title(f"准确率：随训练上升，最终验证 {最佳acc:.0%}"); ax[1].set_xlabel("轮次"); ax[1].set_ylim(0,1.05); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第09课_训练曲线.png",dpi=140,bbox_inches="tight"); plt.close()

fig,a=plt.subplots(figsize=(6.5,5.5))
a.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
a.set_xticks(range(4)); a.set_xticklabels([CN[l] for l in LABELS],rotation=30,ha="right")
a.set_yticks(range(4)); a.set_yticklabels([CN[l] for l in LABELS]); a.set_xlabel("模型预测"); a.set_ylabel("真实")
for i in range(4):
    for j in range(4):
        a.text(j,i,f"{cm[i,j]}\n{cmn[i,j]:.0%}",ha="center",va="center",color="white" if cmn[i,j]>0.5 else "black")
a.set_title("第9课 你训练的小CNN——验证集混淆矩阵")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第09课_验证集混淆矩阵.png",dpi=140,bbox_inches="tight")
print(f"\n训练完成！最佳验证准确率 {最佳acc:.0%}（从零训练、没有预训练的小CNN）")
print("模型已存 models/教学_小CNN_第9课.pt；两张图在 学习/ 目录")
