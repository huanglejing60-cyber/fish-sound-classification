# -*- coding: utf-8 -*-
"""第15课b：结合实验记录重画昼夜曲线（标注放鱼干扰 vs 产卵高峰）"""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

rows=list(csv.DictReader(open(r"E:\大黄鱼分析\学习\第15课_全天扫描结果.csv",encoding="utf-8-sig")))
rel=np.array([float(r["相对小时"]) for r in rows])
seg=np.array([int(r["成片窗口数"]) for r in rows])*3/60.0   # 分钟

# 依据实验记录(3.13)给柱子分类上色
def 颜色(h):
    if 8.8<=h<=9.9:  return "#9ca3af"   # 8:58入水+9:05-9:49放鱼苏醒 → 干扰
    if 18.5<=h<=21.2: return "#1e3a8a"  # 傍晚产卵高峰
    if seg[list(rel).index(h)]>0: return "#60a5fa"  # 其他零星
    return "#cbd5e1"
colors=[颜色(h) for h in rel]

fig,ax=plt.subplots(figsize=(14.5,6))
for lo,hi in [(18,24),(24,30)]:
    ax.axvspan(lo,hi,color="#1e3a8a",alpha=.06)
ax.bar(rel,seg,width=0.42,color=colors,zorder=2)
# 早晨放鱼干扰
ax.axvspan(8.8,9.9,color="#9ca3af",alpha=.18,zorder=0)
ax.text(9.4,23.5,"早晨 8:58–9:50\n入水+打氧+麻醉放鱼270条\n（操作干扰，非鱼声，应排除）",
        ha="center",va="top",fontsize=10,color="#4b5563",
        bbox=dict(boxstyle="round",fc="#f9fafb",ec="#9ca3af"))
# 傍晚产卵高峰
ax.axvspan(19.0,21.0,color="#dc2626",alpha=.08,zorder=0)
ax.text(19.6,23.5,"傍晚 19:00–21:00 产卵高峰\n实验记录：'傍晚六七点叫声明显'\n规则法/CNN/现场观察三方一致",
        ha="center",va="top",fontsize=10,color="#b91c1c",
        bbox=dict(boxstyle="round",fc="#fef2f2",ec="#dc2626"))
ax.text(13.5,20,"白天、深夜均无成片发声",ha="center",fontsize=10,color="#374151",
        bbox=dict(boxstyle="round",fc="#ecfdf5",ec="#10b981"))
ax.set_xlabel("钟点（3月13日 08:51 → 3月14日 08:21）"); ax.set_ylabel("每30分钟成片发声时长（分钟）")
xt=np.arange(8,34,2); ax.set_xticks(xt); ax.set_xticklabels([f"{int(h%24):02d}:00" for h in xt])
ax.set_xlim(8,33); ax.set_ylim(0,25)
ax.set_title("大黄鱼24小时发声曲线（CNN14批量扫描 + 实验记录交叉核对）")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#1e3a8a",label="傍晚产卵"),Patch(facecolor="#9ca3af",label="早晨放鱼干扰"),
                  Patch(facecolor="#60a5fa",label="零星"),Patch(facecolor="#1e3a8a",alpha=.08,label="夜间")],
          loc="upper right",fontsize=9)
ax.grid(alpha=.25,axis="y")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第15课_24小时昼夜曲线_含实验记录.png",dpi=140,bbox_inches="tight")
print("图已生成")
