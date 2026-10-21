# -*- coding: utf-8 -*-
# 第08课补充："步子（学习率）"到底是什么？——把模型简化成只有一个旋钮w
# 旋钮w是一个数字；loss(w)=(w-1.5)^2 是一个碗，碗底(w=1.5)是最优答案。
# 每轮更新： w新 = w旧 - 学习率 * 梯度   （梯度=反向传播算出的"该往哪挪、挪多少"）
# 学习率=每一轮把这个旋钮数字挪动的幅度。下面三种步子，看小球怎么跳。
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

def loss(w): return (w-1.5)**2
def 走(lr,起点=-1.0,步数=30):
    ws=[起点]
    for _ in range(步数):
        w=ws[-1]; 梯度=2*(w-1.5)          # 对loss求导，指向"该往哪走"
        ws.append(w-lr*梯度)             # 真正挪旋钮：挪多少 = 学习率 × 梯度
    return np.array(ws)

w轴=np.linspace(-3.5,4,400)
fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
配置=[(0.05,"小步子 lr=0.05：每轮只挪一点点（慢但稳）",30),
      (0.30,"合适步子 lr=0.3：几步就到碗底",30),
      (1.20,"大步子 lr=1.2：一步跨过头，越弹越远（发散）",12)]
for ax,(lr,title,步数) in zip(axs,配置):
    ws=走(lr,步数=步数)
    ax.plot(w轴,loss(w轴),c="#94a3b8",lw=2)            # 画碗
    ax.plot(ws,loss(ws),"-o",c="#dc2626",ms=6,lw=1.5,alpha=.8)  # 小球每一步
    for i in range(min(len(ws),12)):
        ax.annotate(str(i),(ws[i],loss(ws[i])),textcoords="offset points",xytext=(4,6),fontsize=8,color="#b91c1c")
    ax.axvline(1.5,ls="--",c="#16a34a"); ax.set_title(title,fontsize=12)
    ax.set_xlabel("旋钮 w 的数值"); ax.set_xlim(-3.5,4); ax.set_ylim(0,12)
    print(f"{title[:14]:>16}  旋钮前5步: {np.round(ws[:6],3)}")
axs[0].set_ylabel("loss（错误程度）")
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第08课_步子是什么_单旋钮下山.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("图已保存：",out)
