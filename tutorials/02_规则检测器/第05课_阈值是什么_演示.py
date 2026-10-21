# -*- coding: utf-8 -*-
"""
概念演示：阈值是什么？定高了/定低了会怎样？
用同一段鱼声包络，只改变"门槛高度"，看检出结果的区别。
"""
import numpy as np, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr=16000
y,_=librosa.load(wav,sr=sr,mono=True)
w=sr*10; n=len(y)//w
best,bk=-1,0
for i in range(n):
    s=y[i*w:(i+1)*w]; r=np.sqrt(np.mean(s**2))
    if r>best: best,bk=r,i
y10=y[bk*w:(bk+1)*w]; t=np.arange(len(y10))/sr
env=np.abs(hilbert(y10)); k=int(0.005*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
med=np.median(env); rng=env.max()-med

fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
设置=[(0.25,"阈值定【太高】：只认最响的几声 → 漏掉小声的鱼声（漏检）","#dc2626"),
      (0.06,"阈值【合适】：鱼声基本都抓到，噪声没进来（我们第3课用的）","#16a34a"),
      (0.008,"阈值定【太低】：连水流、小波动都当鱼声 → 误报一大片","#dc2626")]
for ax,(ratio,title,color) in zip(axes,设置):
    thr=med+ratio*rng
    pk,_=find_peaks(env,distance=int(0.02*sr),height=thr,prominence=0.008)
    ax.plot(t,y10,color="#cbd5e1",lw=0.35)
    ax.plot(t,env,color="#1f77b4",lw=1)
    ax.axhline(thr,color=color,ls="--",lw=1.5,label=f"阈值线（{len(pk)}个被判定为鱼声）")
    ax.plot(pk/sr,env[pk],"v",color=color,ms=6)
    ax.set_title(title+f"　→ 检出 {len(pk)} 个",fontsize=12,color=color)
    ax.set_ylabel("振幅"); ax.legend(loc="upper right",fontsize=10); ax.set_xlim(0,10)
axes[-1].set_xlabel("时间(秒)")
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第05课_阈值原理_高低对比.png"
plt.savefig(out,dpi=130,bbox_inches="tight")
print("图已保存：",out)
