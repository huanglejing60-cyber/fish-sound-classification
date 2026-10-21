# -*- coding: utf-8 -*-
"""第17课a（dry-run 核对）：把23个f编号映射到fish目录里的真实文件，
打印每个文件将怎么处理，不实际移动。确认无误后再跑执行脚本。"""
import os, csv, re
BASE=r"E:\大黄鱼分析\dataset_train_v1"
MAN=os.path.join(BASE,"数据划分_manifest.csv")
# 读审查结果（编号->目标类别）
审查={}
with open(os.path.join(BASE,"_待复核标签","审查结果_最终.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        审查[r["编号"]]=r["目标类别"]
# 读待复核清单（编号f01_... -> 钟点/split）
清单={}
with open(os.path.join(BASE,"_待复核标签","待复核清单.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        tag=r["编号"].split("_")[0]
        清单[tag]=(r["钟点"].replace(":",""),r["split"])
# 读manifest的fish行
rows=[]
with open(MAN,encoding="utf-8-sig") as f:
    rd=csv.DictReader(f); rows=list(rd)
fish_rows=[r for r in rows if r["类别"]=="fish"]
print("=== 23个待复核 → 源文件 → 处理（dry-run，未移动）===")
plan=[]
for tag in sorted(审查):
    hm,split=清单[tag]
    cands=[r for r in fish_rows if hm in r["文件名"] and r["split"]==split]
    tgt=审查[tag]
    if len(cands)!=1:
        print(f"[{tag}] 钟点{hm} 匹配到{len(cands)}个！需人工检查"); continue
    src=cands[0]["文件名"]
    if tgt=="fish":
        act="保留 fish（不动）"
    elif tgt=="exclude":
        act=f"隔离 → _隔离剔除/  （从manifest移除）"
    else:
        act=f"改标 → {tgt}/{tgt}/relabel_{tgt}_{src}"
    plan.append((tag,src,tgt,split))
    print(f"{tag} [{split}] {src}\n      {act}")
print("\n=== 移动后类别分布预测 ===")
from collections import Counter
cnt=Counter((r["类别"],r["split"]) for r in rows)
for tag,src,tgt,split in plan:
    if tgt=="fish":continue
    cnt[("fish",split)]-=1
    if tgt=="exclude":continue
    cnt[(tgt,split)]+=1
for cls in ["fish","stress","water","impact"]:
    tr=cnt[(cls,"train")]; va=cnt[(cls,"val")]
    print(f"{cls:7s} train {tr:3d}  val {va:2d}  合计 {tr+va:3d}")
tot=sum(cnt.values())
print("训练集总样本(剔除隔离后)=",tot)
