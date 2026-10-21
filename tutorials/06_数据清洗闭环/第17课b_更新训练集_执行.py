# -*- coding: utf-8 -*-
"""第17课b（执行）：备份manifest→移动文件→重写manifest→验证。"""
import os, csv, shutil
from collections import Counter
BASE=r"E:\大黄鱼分析\dataset_train_v1"
MAN=os.path.join(BASE,"数据划分_manifest.csv")
BAK=os.path.join(BASE,"数据划分_manifest_备份_清洗前.csv")
EXC=os.path.join(BASE,"_隔离剔除"); os.makedirs(EXC,exist_ok=True)
idmap={"fish":0,"stress":1,"water":2,"impact":3}
# 1) 备份manifest
shutil.copy2(MAN,BAK); print("已备份manifest →",os.path.basename(BAK))
# 审查结果 & 清单
审查={}
with open(os.path.join(BASE,"_待复核标签","审查结果_最终.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): 审查[r["编号"]]=r["目标类别"]
清单={}
with open(os.path.join(BASE,"_待复核标签","待复核清单.csv"),encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): 清单[r["编号"].split("_")[0]]=(r["钟点"].replace(":",""),r["split"])
with open(MAN,encoding="utf-8-sig") as f:
    rd=csv.DictReader(f); fields=rd.fieldnames; rows=list(rd)
fish_rows=[r for r in rows if r["类别"]=="fish"]
newrows=[]; n_move=0; n_exc=0; n_keep=0
for r in rows:
    if r["类别"]!="fish":
        newrows.append(r); continue
    # 找这个fish是否属于23个待复核
    hit=None
    for tag,(hm,split) in 清单.items():
        if hm in r["文件名"] and split==r["split"] and tag in 审查:
            hit=tag; break
    if hit is None:
        newrows.append(r); continue
    tgt=审查[hit]
    src=os.path.join(BASE,"fish",r["文件名"])
    if tgt=="fish":
        newrows.append(r); n_keep+=1
    elif tgt=="exclude":
        shutil.move(src,os.path.join(EXC,f"{hit}_{r['文件名']}")); n_exc+=1
    else:
        newname=f"relabel_{tgt}_{r['文件名']}"
        shutil.move(src,os.path.join(BASE,tgt,newname))
        r["文件名"]=newname; r["类别"]=tgt; r["类别id"]=idmap[tgt]
        r["来源"]="人工复核改标:"+r["来源"]
        newrows.append(r); n_move+=1
# 重写manifest
with open(MAN,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(newrows)
print(f"文件处理：保留fish {n_keep} 个，改标移动 {n_move} 个，隔离剔除 {n_exc} 个")
# 验证：目录文件数 + manifest分布
print("\n=== 目录实际文件数 ===")
for cls in ["fish","stress","water","impact"]:
    print(cls, len([x for x in os.listdir(os.path.join(BASE,cls)) if x.endswith(".wav")]))
print("隔离剔除", len([x for x in os.listdir(EXC) if x.endswith(".wav")]))
print("\n=== 新manifest分布 ===")
for cls in ["fish","stress","water","impact"]:
    tr=sum(1 for r in newrows if r["类别"]==cls and r["split"]=="train")
    va=sum(1 for r in newrows if r["类别"]==cls and r["split"]=="val")
    print(f"{cls:7s} train {tr:3d} val {va:2d} 合计 {tr+va:3d}")
print("manifest总行数:",len(newrows))
