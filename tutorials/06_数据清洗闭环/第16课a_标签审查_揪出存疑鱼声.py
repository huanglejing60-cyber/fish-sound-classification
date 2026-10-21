# -*- coding: utf-8 -*-
"""第16课a：数据闭环第1步——标签审查。
找出训练集里"非产卵时段(18:00-21:30)却被标成fish"的样本，
复制到待复核文件夹并画语谱图，供人工耳听判定。"""
import os,re,csv,shutil
import numpy as np, librosa, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False

ROOT=r"E:\大黄鱼分析\dataset_train_v1"
MAN=os.path.join(ROOT,"数据划分_manifest.csv")
OUT=os.path.join(ROOT,"_待复核标签")
os.makedirs(OUT,exist_ok=True)
SR=32000

def 钟点(文件名,源段):
    """尽力解析样本真实钟点（小时.分钟）"""
    m=re.search(r'_(\d{6})_段',文件名) or re.search(r'真实(\d{6})',文件名) or re.search(r'_(\d{6})_脉冲',文件名)
    if m:
        t=m.group(1); return int(t[:2])+int(t[2:4])/60+int(t[4:6])/3600, f"{t[:2]}:{t[2:4]}:{t[4:6]}"
    # v2_XX分YY秒：源段起始 + 分秒
    m=re.search(r'(\d{2})分(\d{2})秒',文件名)
    if m and 源段:
        base=int(源段[:2])*3600+int(源段[2:4])*60+int(源段[4:6])
        sec=base+int(m.group(1))*60+int(m.group(2))
        return sec/3600, f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"
    return None,"?"

def 产卵窗(h):
    return h is not None and 18.0<=h<=21.5

rows=list(csv.DictReader(open(MAN,encoding="utf-8-sig")))
fish=[r for r in rows if r["类别"]=="fish"]
待复核=[]
for r in fish:
    h,txt=钟点(r["文件名"],r["源段"])
    if not 产卵窗(h):   # 非产卵时段却标 fish
        待复核.append((r["文件名"],r["split"],r["来源"],h,txt))
待复核.sort(key=lambda x:(x[3] if x[3] is not None else 99))

print(f"fish总数={len(fish)}，其中非产卵时段(18:00-21:30)待复核={len(待复核)}")
清单=[]
for i,(fn,split,src,h,txt) in enumerate(待复核,1):
    srcpath=os.path.join(ROOT,"fish",fn)
    if not os.path.exists(srcpath):
        print("缺失:",fn); continue
    tag=f"f{i:02d}_{txt.replace(':','')}_{split}"
    dst=os.path.join(OUT,tag+".wav")
    shutil.copy(srcpath,dst)
    清单.append((tag,txt,split,src))
    print(f"{tag}  钟点{txt}  [{split}/{src}]")

# 清单csv
with open(os.path.join(OUT,"待复核清单.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["编号","钟点","split","来源","你的判定(鱼/应激/水/撞击/删除)"])
    for tag,txt,split,src in 清单: w.writerow([tag,txt,split,src,""])

# 画语谱网格
n=len(清单)
if n:
    cols=4; rowsN=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rowsN,cols,figsize=(cols*4.2,rowsN*3))
    axes=np.array(axes).reshape(-1)
    for i,(tag,txt,split,src) in enumerate(清单):
        ax=axes[i]
        y,sr=sf.read(os.path.join(OUT,tag+".wav"))
        if y.ndim>1: y=y.mean(1)
        if sr!=SR: y=librosa.resample(y,orig_sr=sr,target_sr=SR)
        M=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=64,fmin=50,fmax=14000)
        D=librosa.power_to_db(M,ref=1.0,amin=1e-10)
        ax.imshow(D,origin="lower",aspect="auto",cmap="magma",extent=[0,3,50,14000])
        ax.set_title(f"{tag}  {txt}  [{split[0]}]",fontsize=10)
        ax.set_xlabel("秒"); ax.set_ylabel("Hz")
        ax.tick_params(labelsize=7)
    for j in range(n,len(axes)): axes[j].axis("off")
    plt.suptitle("待复核：非产卵时段被标成fish的样本（耳听确认是不是真鱼）",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"待复核语谱图.png"),dpi=130,bbox_inches="tight")
    print("图：",os.path.join(OUT,"待复核语谱图.png"))
