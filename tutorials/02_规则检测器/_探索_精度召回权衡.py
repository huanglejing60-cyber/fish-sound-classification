# -*- coding: utf-8 -*-
# 画"精度-召回跷跷板"图（数据来自上一步扫描）
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
T=[1,2,3,4,5,6]
精度=[73,100,100,100,100,0]
召回=[38,17,5,2,2,0]
块数=[22,7,2,1,1,0]
fig,ax=plt.subplots(figsize=(10,5.5))
ax.plot(T,精度,"o-",color="#16a34a",lw=2.5,ms=9,label="精度 Precision（报得准不准）")
ax.plot(T,召回,"s-",color="#dc2626",lw=2.5,ms=9,label="召回 Recall（抓得全不全）")
for i in range(len(T)):
    ax.annotate(f"{块数[i]}块",(T[i],召回[i]),textcoords="offset points",xytext=(0,-16),
                ha="center",color="#dc2626",fontsize=10)
ax.axvspan(1.7,2.3,color="#fde68a",alpha=0.6)
ax.text(2,58,"甜点区\n（准又没漏太多）",ha="center",color="#b45309",fontsize=11)
ax.annotate("阈值太大→什么都不报\n精度再高也没意义（检测器失效）",xy=(6,0),xytext=(4.2,35),
            fontsize=11,color="#7f1d1d",arrowprops=dict(arrowstyle="->",color="#7f1d1d"))
ax.set_xlabel("组数阈值（每组10秒，要求有几对成声才算鱼）"); ax.set_ylabel("百分比 %")
ax.set_title("精度和召回是'跷跷板'：不是阈值越大越好，要找平衡点",fontsize=13)
ax.set_xticks(T); ax.set_ylim(-5,110); ax.legend(fontsize=11,loc="center right"); ax.grid(alpha=.3)
plt.tight_layout()
out=r"E:\大黄鱼分析\学习\第05课_精度召回跷跷板.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("已存",out)
