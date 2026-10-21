# -*- coding: utf-8 -*-
"""第19课d2：量化调优四版本对比图（只加载已生成模型评估，不重新量化）"""
import os, csv
import numpy as np, librosa, torch
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL_DIR=r"E:\大黄鱼分析\models"
V2=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt")
版本=[("fp32\n原版",os.path.join(MODEL_DIR,"cnn14_v2.onnx")),
      ("v1\n朴素INT8",os.path.join(MODEL_DIR,"cnn14_v2_int8.onnx")),
      ("v2\nEntropy校准",os.path.join(MODEL_DIR,"cnn14_v2_int8_entropy.onnx")),
      ("v3\n混合精度",os.path.join(MODEL_DIR,"cnn14_v2_int8_mix.onnx"))]
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
def 读波形(p):
    y,_=librosa.load(p,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]; return y
def 转mel(y):
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
ck=torch.load(V2,map_location="cpu",weights_only=False); mean,std=ck["mean"],ck["std"]
va=[r for r in csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")) if r["split"]=="val"]
Xva=np.stack([(转mel(读波形(os.path.join(ROOT,r["类别"],r["文件名"])))-mean)/std for r in va]).astype(np.float32)
Yva=np.array([int(r["类别id"]) for r in va])
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def 评估(path):
    s=ort.InferenceSession(path,providers=["CPUExecutionProvider"]); pr=softmax(s.run(["logits"],{"mel":Xva})[0]).argmax(1)
    acc=(pr==Yva).mean(); 鱼=Yva==0
    TP=((pr==0)&鱼).sum(); FP=((pr==0)&~鱼).sum(); FN=((pr!=0)&鱼).sum()
    return acc*100, TP/max(TP+FP,1)*100, TP/max(TP+FN,1)*100, os.path.getsize(path)/1e6
D=[(name,)+评估(path) for name,path in 版本]
names=[d[0] for d in D]; sizes=[d[4] for d in D]; acc=[d[1] for d in D]; prec=[d[2] for d in D]; rec=[d[3] for d in D]
cols=["#2563eb","#94a3b8","#60a5fa","#16a34a"]
fig,ax=plt.subplots(2,2,figsize=(12,8))
def bar(a,v,t,unit,fmt="{:.0f}"):
    b=a.bar(names,v,color=cols)
    for r,x in zip(b,v): a.text(r.get_x()+r.get_width()/2,x+max(v)*0.02,fmt.format(x)+unit,ha="center",fontsize=10)
    a.set_title(t); a.grid(alpha=.3,axis="y")
bar(ax[0,0],sizes,"模型体积（越小越好）","MB")
bar(ax[0,1],acc,"验证准确率（越高越好）","%")
bar(ax[1,0],prec,"真鱼精度（导师红线：越高越好）","%")
bar(ax[1,1],rec,"真鱼召回","%")
plt.suptitle("第19课d 量化调优：混合精度(v3)在体积压到1/4的同时，精度/精度/召回全部回到fp32水平",fontsize=13)
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第19课d_量化调优对比.png",dpi=140,bbox_inches="tight")
print("图已生成")
