# -*- coding: utf-8 -*-
"""
=====================================================================
第 19 课 d：把量化掉的精度"调回来"（量化调优，部署工程师的日常）
---------------------------------------------------------------------
三个版本横向对比：
  v1 MinMax校准+32样本（上一版，掉精度）
  v2 Entropy(KL散度)校准+全部训练样本（校准更鲁棒）
  v3 在v2基础上，把最敏感的"第一层卷积 + 最后分类头"保留fp32（混合精度）
看谁能在"体积/速度/精度"之间取得最好平衡。
=====================================================================
"""
import os, csv, time, tempfile, shutil
import numpy as np, librosa, torch
import onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL_DIR=r"E:\大黄鱼分析\models"
V2=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt"); ONNX=os.path.join(MODEL_DIR,"cnn14_v2.onnx")
V1=os.path.join(MODEL_DIR,"cnn14_v2_int8.onnx")
V2Q=os.path.join(MODEL_DIR,"cnn14_v2_int8_entropy.onnx")
V3Q=os.path.join(MODEL_DIR,"cnn14_v2_int8_mix.onnx")
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
    return np.stack([((转mel(读波形(os.path.join(ROOT,r["类别"],r["文件名"]))))-mean)/std for r in sub]).astype(np.float32)
Xcal=准备(tr); Xva=准备(va); Yva=np.array([int(r["类别id"]) for r in va])
print(f"校准样本 {len(Xcal)} 个，验证 {len(va)} 个")

tmp=tempfile.mkdtemp(prefix="onnxq_"); src=os.path.join(tmp,"m.onnx"); shutil.copy(ONNX,src)
class Cal(CalibrationDataReader):
    def __init__(self,X): self.en=iter([{"mel":X[i:i+1]} for i in range(len(X))])
    def get_next(self):
        try: return next(self.en)
        except StopIteration: return None
def 量化(出,方法,排除=None):
    t=os.path.join(tmp,os.path.basename(出))
    quantize_static(src,t,Cal(Xcal),quant_format=QuantFormat.QDQ,per_channel=True,
                    weight_type=QuantType.QInt8,activation_type=QuantType.QUInt8,
                    calibrate_method=方法,nodes_to_exclude=排除)
    shutil.copy(t,出)
敏感层=["/m/conv_block1/conv1/Conv","/m/conv_block1/conv2/Conv","/m/fc_audioset/Gemm"]
print("生成 v2 Entropy 校准（样本多，约1-2分钟）...")
量化(V2Q,CalibrationMethod.Entropy)
print("生成 v3 Entropy + 敏感层保留fp32 ...")
量化(V3Q,CalibrationMethod.Entropy,排除=敏感层)

def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def 评估(path):
    s=ort.InferenceSession(path,providers=["CPUExecutionProvider"])
    p=softmax(s.run(["logits"],{"mel":Xva})[0]); pr=p.argmax(1)
    acc=(pr==Yva).mean(); 鱼m=Yva==0
    TP=((pr==0)&鱼m).sum(); FP=((pr==0)&~鱼m).sum(); FN=((pr!=0)&鱼m).sum()
    return acc,TP/max(TP+FP,1),TP/max(TP+FN,1),p
def 测(path,bs,n=30,预热=5):
    s=ort.InferenceSession(path,providers=["CPUExecutionProvider"]); x=np.random.randn(bs,N_MELS,N_FRAMES).astype(np.float32)
    for _ in range(预热): s.run(["logits"],{"mel":x})
    t=[]
    for _ in range(n):
        t0=time.perf_counter(); s.run(["logits"],{"mel":x}); t.append((time.perf_counter()-t0)*1000)
    return np.median(t)
版本=[("fp32 原版",ONNX),("v1 MinMax+32",V1),("v2 Entropy+全量",V2Q),("v3 Entropy+混合精度",V3Q)]
print("\n版本                    体积MB  准确率  鱼精度  鱼召回  b16延迟ms")
for name,path in 版本:
    a,prec,rec,_=评估(path); print(f"{name:22s} {os.path.getsize(path)/1e6:6.0f}  {a:6.0%}  {prec:6.0%}  {rec:6.0%}  {测(path,16):7.1f}")
p32=评估(ONNX)[3]; p2=评估(V2Q)[3]; p3=评估(V3Q)[3]
print(f"\n概率最大漂移 vs fp32：v2={np.abs(p32-p2).max():.3f}  v3={np.abs(p32-p3).max():.3f}")
print("v3与fp32分类一致：", int((p32.argmax(1)==p3.argmax(1)).sum()), "/", len(va))
