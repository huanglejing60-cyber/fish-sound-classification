# -*- coding: utf-8 -*-
"""
=====================================================================
第 08 课：模型是怎么"学会"的？——用一个最小神经网络，看它现场学习
---------------------------------------------------------------------
我们把"区分鱼声/噪声"先压缩成最简单的样子：
  每段声音只看两个数（两个特征）：
     x1 = 脉冲间隔（归一化，鱼约90-130ms）
     x2 = 峰频（归一化，鱼约800Hz）
  模型要学的：在这张"特征图"上画一条分界线，把红点(鱼)和蓝点(噪声)分开。
训练就是：一开始线乱画、错很多 → 不断根据"错得多惨"微调旋钮 → 线越来越准。
图会画出：训练不同阶段(0/10/40/100/300轮)分界线长啥样，以及错误率(loss)怎么下降。
=====================================================================
"""
import numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
torch.manual_seed(0); np.random.seed(0)
DEVICE="cpu"   # 这个玩具网络太小，用CPU就行，重点是看懂原理

# ---------- 1. 造数据（用两个特征，红点=鱼，蓝点=噪声）----------
N=120
# 鱼：间隔短、峰频低而集中（左下一团）
鱼=np.random.randn(N,2)*np.array([0.09,0.09])+np.array([0.30,0.32])
# 噪声：间隔杂、峰频高而分散（右上一团，更散）
噪=np.random.randn(N,2)*np.array([0.16,0.15])+np.array([0.72,0.70])
X=np.vstack([鱼,噪]).astype(np.float32)
Y=np.array([0]*N+[1]*N)   # 0=鱼，1=噪声
Xt=torch.from_numpy(X); Yt=torch.from_numpy(Y)

# ---------- 2. 模型：一个极小的神经网络（旋钮就是里面的权重）----------
# 结构：2个输入 → 16个"神经元" → 2个输出(鱼/噪声)。就是几层加权求和。
模型=nn.Sequential(
    nn.Linear(2,16), nn.ReLU(),
    nn.Linear(16,2)
)
# 损失函数：衡量"预测和标准答案差多远"（错得越离谱，loss越大）
损失=nn.CrossEntropyLoss()
# 优化器：根据loss，反着调旋钮的"手"。lr=学习率=每次调整的步子大小
优化器=torch.optim.Adam(模型.parameters(), lr=1.5)

# 画分界线背景用的网格
gx,gy=np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200))
网格=torch.from_numpy(np.c_[gx.ravel(),gy.ravel()].astype(np.float32))
def 画背景(ax,模型):
    with torch.no_grad():
        z=模型(网格).argmax(1).numpy().reshape(gx.shape)
    ax.contourf(gx,gy,z,levels=[-0.5,0.5,1.5],colors=["#fecaca","#bfdbfe"],alpha=0.6)

# ---------- 3. 训练循环（深度学习的核心就这5行，反复做）----------
快照轮次=[0,10,40,100,300]; 快照={}
loss曲线=[]; acc曲线=[]
for 轮 in range(301):
    if 轮 in 快照轮次: 快照[轮]={k:v.clone() for k,v in 模型.state_dict().items()}
    优化器.zero_grad()          # ① 把上一轮旋钮的调整量清零
    logits=模型(Xt)             # ② 前向：用现在的旋钮，预测每个点是鱼还是噪声
    l=损失(logits,Yt)          # ③ 打分：和标准答案比，得到loss(错得多惨)
    l.backward()               # ④ 反向：算出每个旋钮该往哪个方向调、调多少
    优化器.step()              # ⑤ 更新：真正动手把所有旋钮拧一小下
    with torch.no_grad():
        acc=(logits.argmax(1).numpy()==Y).mean()
    loss曲线.append(l.item()); acc曲线.append(acc*100)

# ---------- 4. 画图：分界线学习过程 + loss/准确率曲线 ----------
fig=plt.figure(figsize=(15,8))
gs=fig.add_gridspec(2,5,height_ratios=[1.2,1])
for i,ep in enumerate(快照轮次):
    ax=fig.add_subplot(gs[0,i])
    模型.load_state_dict(快照[ep]); 画背景(ax,模型)
    ax.scatter(鱼[:,0],鱼[:,1],c="#dc2626",s=12,label="鱼"); ax.scatter(噪[:,0],噪[:,1],c="#2563eb",s=12,label="噪声")
    ax.set_title(f"训练第 {ep} 轮\n(准确率 {acc曲线[ep]:.0f}%, loss {loss曲线[ep]:.2f})",fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
    if i==0: ax.legend(loc="upper left",fontsize=9)
ax1=fig.add_subplot(gs[1,0:3]); ax1.plot(loss曲线,c="#dc2626",lw=2); ax1.set_title("loss（错误程度）随训练不断下降",fontsize=12)
ax1.set_xlabel("训练轮次 epoch"); ax1.set_ylabel("loss 越小越好"); ax1.grid(alpha=.3)
ax2=fig.add_subplot(gs[1,3:5]); ax2.plot(acc曲线,c="#16a34a",lw=2); ax2.set_title("准确率随训练不断上升",fontsize=12)
ax2.set_xlabel("训练轮次 epoch"); ax2.set_ylabel("准确率 %"); ax2.set_ylim(0,105); ax2.grid(alpha=.3)
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第08课_模型学习过程_分界线与loss.png"
plt.savefig(out,dpi=130,bbox_inches="tight")
print("第0轮:  loss %.3f 准确率 %.0f%%（旋钮随机，基本瞎猜）"%(loss曲线[0],acc曲线[0]))
print("第300轮:loss %.3f 准确率 %.0f%%（训练后，几乎全对）"%(loss曲线[-1],acc曲线[-1]))
print("图已保存：",out)
