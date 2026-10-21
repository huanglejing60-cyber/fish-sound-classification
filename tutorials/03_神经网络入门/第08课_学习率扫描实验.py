# -*- coding: utf-8 -*-
# 第08课练习：扫描不同学习率，看"步子大小"如何影响训练（同一道题、相同起点，保证公平）
import numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
np.random.seed(0)
N=120
鱼=np.random.randn(N,2)*np.array([0.09,0.09])+np.array([0.30,0.32])
噪=np.random.randn(N,2)*np.array([0.16,0.15])+np.array([0.72,0.70])
X=torch.from_numpy(np.vstack([鱼,噪]).astype(np.float32)); Y=torch.from_numpy(np.array([0]*N+[1]*N))

def 建模型():
    torch.manual_seed(42)   # 每个学习率都从"同一套随机旋钮"开始，公平对比
    return nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,2))

def 跑(lr):
    m=建模型(); opt=torch.optim.Adam(m.parameters(),lr=lr); crit=nn.CrossEntropyLoss()
    ls=[]; ac=[]
    for _ in range(300):
        opt.zero_grad(); lo=m(X); l=crit(lo,Y); l.backward(); opt.step()
        ls.append(l.item()); ac.append((lo.argmax(1).numpy()==Y.numpy()).mean()*100)
    return ls,ac

学习率=[0.0003,0.003,0.03,0.3,3,30]
颜色=plt.cm.viridis(np.linspace(0,0.92,len(学习率)))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
print("学习率   最终loss  最终准确率")
for lr,c in zip(学习率,颜色):
    ls,ac=跑(lr)
    ax1.plot(ls,color=c,lw=2,label=f"lr={lr}")
    ax2.plot(ac,color=c,lw=2,label=f"lr={lr}")
    print(f"{lr:<7} {ls[-1]:>8.3f}  {ac[-1]:>6.0f}%")
ax1.set_ylim(0,2); ax1.set_title("不同学习率的 loss 曲线（越稳、越低越好）"); ax1.set_xlabel("训练轮次"); ax1.set_ylabel("loss")
ax2.set_ylim(0,105); ax2.set_title("不同学习率的准确率曲线"); ax2.set_xlabel("训练轮次"); ax2.set_ylabel("准确率 %")
for ax in (ax1,ax2): ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第08课_学习率扫描.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("图已保存：",out)
