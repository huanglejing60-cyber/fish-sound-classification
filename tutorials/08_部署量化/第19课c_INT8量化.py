# -*- coding: utf-8 -*-
"""
=====================================================================
第 19 课 c：静态量化（INT8）——把模型压到约1/4、CPU更快，再查精度
---------------------------------------------------------------------
fp32 权重占4字节；INT8 用8位整数近似（占1字节）。卷积层占本模型大头，
所以用"静态量化"（拿少量真实样本校准），这是工业界标准做法。
对比 fp32 vs INT8：体积 / 验证精度 / 延迟 / 概率漂移。
=====================================================================
"""
import os, csv, time, tempfile, shutil
import numpy as np, librosa, torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL_DIR=r"E:\大黄鱼分析\models"
V2=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt"); ONNX=os.path.join(MODEL_DIR,"cnn14_v2.onnx")
STA=os.path.join(MODEL_DIR,"cnn14_v2_int8.onnx")
LABELS=["fish","stress","water","impact"]
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300
def 读波形(p):
    y,_=librosa.load(p,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]; return y
def 转mel(y):
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]
ck=torch.load(V2,map_location="cpu",weights_only=False); mean,std=ck["mean"],ck["std"]
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
tr=[r for r in rows if r["split"]=="train"]; va=[r for r in rows if r["split"]=="val"]
def 准备(sub):
    X=[]
    for r in sub:
        X.append(((转mel(读波形(os.path.join(ROOT,r["类别"],r["文件名"]))))-mean)/std)
    return np.stack(X).astype(np.float32)
Xcal=准备(tr[:32]); Xva=准备(va); Yva=np.array([int(r["类别id"]) for r in va])

# 静态量化（中文路径会让量化工具找不到临时文件，放英文临时目录做完再复制回来）
tmp=tempfile.mkdtemp(prefix="onnxq_")
src=os.path.join(tmp,"model.onnx"); sta_t=os.path.join(tmp,"sta.onnx"); shutil.copy(ONNX,src)
class Cal(CalibrationDataReader):
    def __init__(self,X): self.en=iter([{"mel":X[i:i+1]} for i in range(len(X))])
    def get_next(self):
        try: return next(self.en)
        except StopIteration: return None
quantize_static(src,sta_t,Cal(Xcal),quant_format=QuantFormat.QDQ,per_channel=True,
                weight_type=QuantType.QInt8,activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax)
shutil.copy(sta_t,STA)
print("已生成 INT8 静态量化模型")

def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def 评估(path):
    s=ort.InferenceSession(path,providers=["CPUExecutionProvider"])
    p=softmax(s.run(["logits"],{"mel":Xva})[0]); pr=p.argmax(1)
    acc=(pr==Yva).mean(); 鱼m=Yva==0
    TP=((pr==0)&鱼m).sum(); FP=((pr==0)&~鱼m).sum(); FN=((pr!=0)&鱼m).sum()
    return acc,TP/max(TP+FP,1),TP/max(TP+FN,1),p
def 测(path,bs,n=40,预热=6):
    s=ort.InferenceSession(path,providers=["CPUExecutionProvider"]); x=np.random.randn(bs,N_MELS,N_FRAMES).astype(np.float32)
    for _ in range(预热): s.run(["logits"],{"mel":x})
    t=[]
    for _ in range(n):
        t0=time.perf_counter(); s.run(["logits"],{"mel":x}); t.append((time.perf_counter()-t0)*1000)
    return np.median(t)

models=[("fp32 原版",ONNX),("INT8 静态量化",STA)]
print("\n模型              体积(MB)  验证准确率  真鱼精度  真鱼召回  延迟b1  延迟b16")
统计=[]
for name,path in models:
    a,prec,rec,p=评估(path); d1=测(path,1); d16=测(path,16); size=os.path.getsize(path)/1e6
    统计.append((name,size,a,prec,rec,d1,d16))
    print(f"{name:14s} {size:7.1f}   {a:7.0%}    {prec:6.0%}    {rec:6.0%}  {d1:6.1f}  {d16:7.1f}")
p32=评估(ONNX)[3]; psta=评估(STA)[3]
print(f"\n【量化误差】fp32 vs INT8 概率最大漂移 = {np.abs(p32-psta).max():.4f}（<0.05 可接受）")
print("分类结果完全一致：", int((p32.argmax(1)==psta.argmax(1)).sum()), "/", len(va))
print(f"体积压缩：{os.path.getsize(ONNX)/os.path.getsize(STA):.2f} 倍")

names=[s[0] for s in 统计]; sizes=[s[1] for s in 统计]; accs=[s[2]*100 for s in 统计]; d16=[s[6] for s in 统计]
fig,ax=plt.subplots(1,3,figsize=(14,4.5))
for a,v,t,unit in [(ax[0],sizes,"模型体积","MB"),(ax[1],d16,"batch16延迟","ms"),(ax[2],accs,"验证准确率","%")]:
    bars=a.bar(names,v,color=["#2563eb","#16a34a"])
    for b,x in zip(bars,v): a.text(b.get_x()+b.get_width()/2,x+max(v)*0.02,f"{x:.0f}{unit}",ha="center",fontsize=11)
    a.set_title(t); a.grid(alpha=.3,axis="y")
plt.suptitle("第19课c INT8静态量化：体积约1/4、CPU更快、准确率是否保持")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第19课c_量化对比.png",dpi=140,bbox_inches="tight")
print("图在 学习/第19课c_量化对比.png")
