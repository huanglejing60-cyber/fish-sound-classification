# -*- coding: utf-8 -*-
"""
=====================================================================
第 14 课：自动画产卵曲线——平滑 + 成片判定 + 交叉验证
---------------------------------------------------------------------
第13课发现：单看3秒窗口，会有零星孤立尖峰(可能是误报/单声)。
真正的产卵是"连续成片的咕咕"。用两招去伪存真：
  1) 平滑：对鱼概率做1分钟滑动平均——孤立尖峰被摊薄，成片高峰被保留；
  2) 成片判定：找"连续(允许短暂停顿)超过阈值、且持续足够久"的时间段，
     零星1-2个窗口的不报告。
最后和规则法(四道关)得到的23~29分钟高峰交叉验证。
=====================================================================
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

rows=list(csv.DictReader(open(r"E:\大黄鱼分析\学习\第13课_30分钟扫描结果.csv",encoding="utf-8-sig")))
t=np.array([float(r["中点秒"]) for r in rows])/60.0      # 分钟
p=np.array([float(r["鱼概率"]) for r in rows])

# ---------- 1) 1分钟居中滑动平均（20个3秒窗口）----------
def moving(x,w=20):
    k=np.ones(w)/w
    return np.convolve(x,k,mode="same")
s=moving(p,20)

# ---------- 2) 找成片段：平滑值≥阈值，允许≤1分钟小停顿，总时长≥1.5分钟 ----------
def 成片段(s,t,thr=0.30,gap=20,min_len=30):
    idx=np.where(s>=thr)[0]
    if len(idx)==0: return []
    粗=[]; start=idx[0]; prev=idx[0]
    for i in idx[1:]:
        if i-prev<=gap: prev=i
        else: 粗.append((start,prev)); start=i; prev=i
    粗.append((start,prev))
    return [(a,b) for a,b in 粗 if b-a>=min_len]
段=成片段(s,t)

print("="*60)
print("CNN14 自动检测到的【成片产卵时段】（平滑值≥0.30且连续≥1.5分钟）：")
for a,b in 段:
    print(f"  {t[a]:.1f} ~ {t[b]:.1f} 分钟（持续约 {t[b]-t[a]:.1f} 分钟，峰值 {s[a:b+1].max():.2f}）")
if not 段: print("  无")
print("规则法(四道关)此前判定：23 ~ 29 分钟")
print("="*60)

# ---------- 3) 画图 ----------
fig,ax=plt.subplots(figsize=(13.5,5.8))
ax.plot(t,p,color="#cbd5e1",lw=0.8,zorder=1,label="原始3秒窗口鱼概率（含零星尖峰）")
ax.plot(t,s,color="#2563eb",lw=2.6,zorder=3,label="1分钟平滑曲线（产卵密度）")
ax.axhline(0.30,ls="--",color="#dc2626",lw=1.2,zorder=2,label="成片阈值0.30")
for a,b in 段:
    ax.axvspan(t[a],t[b],color="#16a34a",alpha=.15,zorder=0)
ax.axvspan(23,29,color="#f59e0b",alpha=.10,zorder=0)
ax.axvline(23,color="#d97706",ls=":",lw=1.5); ax.axvline(29,color="#d97706",ls=":",lw=1.5)
ax.text(26,0.96,"规则法高峰 23~29 分钟",color="#b45309",ha="center",fontsize=10,
        bbox=dict(boxstyle="round",fc="#fffbeb",ec="#f59e0b"))
for a,b in 段:
    ax.text((t[a]+t[b])/2,0.88,"CNN成片高峰",color="#15803d",ha="center",fontsize=10,
            bbox=dict(boxstyle="round",fc="#f0fdf4",ec="#16a34a"))
ax.set_xlabel("时间（分钟）"); ax.set_ylabel("鱼声水平（鱼概率）"); ax.set_ylim(0,1)
ax.set_xlim(0,30); ax.set_title("第14课 大黄鱼产卵声曲线（CNN14滑窗+平滑）：自动定位成片高峰")
ax.legend(loc="upper left",fontsize=9); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第14课_产卵曲线.png",dpi=140,bbox_inches="tight")
print("图在 学习/第14课_产卵曲线.png")
