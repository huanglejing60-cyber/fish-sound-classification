# -*- coding: utf-8 -*-
"""
=====================================================================
第 19 课 a：把训练好的 CNN14 导出成 ONNX，并验证"导出后结果没变"
---------------------------------------------------------------------
两步走：
  1) 用 torch.onnx.export 把模型"翻译"成通用的 .onnx 文件（只需一次）
  2) 用 ONNX Runtime 加载它，喂同一个输入，和 PyTorch 的输出逐个对比
     —— 部署的第一铁律：导出后不能算错，数值必须对得上。
本脚本全程在 CPU 上做（产品端通常就是 CPU/芯片）。
=====================================================================
"""
import os, csv
import numpy as np, librosa, torch, torch.nn as nn, torch.nn.functional as F
import onnx, onnxruntime as ort
ROOT=r"E:\大黄鱼分析\dataset_train_v1"; MODEL_DIR=r"E:\大黄鱼分析\models"
V2=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt"); ONNX=os.path.join(MODEL_DIR,"cnn14_v2.onnx")
LABELS=["fish","stress","water","impact"]
SR=32000; N_FFT=1024; HOP=320; N_MELS=64; FMIN=50; FMAX=14000; N_FRAMES=300

def init_layer(l):
    if isinstance(l,(nn.Linear,nn.Conv2d)):
        nn.init.xavier_uniform_(l.weight)
        if l.bias is not None: l.bias.data.fill_(0.)
def init_bn(bn): bn.bias.data.fill_(0.); bn.weight.data.fill_(1.)
class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.conv1=nn.Conv2d(ci,co,3,padding=1,bias=False); self.conv2=nn.Conv2d(co,co,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(co); self.bn2=nn.BatchNorm2d(co)
        init_layer(self.conv1); init_layer(self.conv2); init_bn(self.bn1); init_bn(self.bn2)
    def forward(self,x):
        x=F.relu_(self.bn1(self.conv1(x))); x=F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x,kernel_size=(2,2),stride=(2,2))
class Cnn14(nn.Module):
    def __init__(self,classes=4):
        super().__init__()
        self.bn0=nn.BatchNorm2d(N_MELS)
        self.conv_block1=ConvBlock(1,64); self.conv_block2=ConvBlock(64,128); self.conv_block3=ConvBlock(128,256)
        self.conv_block4=ConvBlock(256,512); self.conv_block5=ConvBlock(512,1024); self.conv_block6=ConvBlock(1024,2048)
        self.fc1=nn.Linear(2048,2048); self.fc_audioset=nn.Linear(2048,classes)
        init_bn(self.bn0); init_layer(self.fc1); init_layer(self.fc_audioset)
    def forward(self,x):
        x=x.unsqueeze(1).permute(0,2,3,1); x=self.bn0(x); x=x.permute(0,3,2,1)
        x=self.conv_block1(x); x=self.conv_block2(x); x=self.conv_block3(x)
        x=self.conv_block4(x); x=self.conv_block5(x); x=self.conv_block6(x)
        x=torch.mean(x,dim=3); x1,_=torch.max(x,dim=2); x=x1+torch.mean(x,dim=2)
        x=F.dropout(x,0.5,self.training); x=F.relu_(self.fc1(x)); emb=F.dropout(x,0.5,self.training)
        return self.fc_audioset(emb),emb
# 部署专用外壳：只输出 logits（上线不需要 emb 那个中间向量）
class 部署模型(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,x):
        logits,_=self.m(x); return logits
def 读波形(path):
    y,_=librosa.load(path,sr=SR,mono=True); y=np.pad(y,(0,max(0,SR*3-len(y))))[:SR*3]; return y
def 转mel(y):
    S=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX)
    S=librosa.power_to_db(S,ref=1.0,amin=1e-10,top_db=None).astype(np.float32)
    return np.pad(S,((0,0),(0,max(0,N_FRAMES-S.shape[1]))))[:,:N_FRAMES]

# ---------- 1. 加载训练好的模型 ----------
ck=torch.load(V2,map_location="cpu",weights_only=False)
mean,std=ck["mean"],ck["std"]
m=Cnn14(4); m.load_state_dict(ck["state_dict"]); m.eval()
em=部署模型(m).eval()
print("已加载 v2_clean 模型")

# ---------- 2. 导出 ONNX ----------
dummy=torch.randn(1,N_MELS,N_FRAMES,dtype=torch.float32)
torch.onnx.export(
    em,dummy,ONNX,
    input_names=["mel"],output_names=["logits"],
    opset_version=17,do_constant_folding=True,dynamo=False,
    dynamic_axes={"mel":{0:"batch"},"logits":{0:"batch"}})  # 支持批量
onnx.checker.check_model(onnx.load(ONNX))
print(f"已导出并通过检查：{os.path.basename(ONNX)}（{os.path.getsize(ONNX)/1e6:.1f} MB）")

# ---------- 3. ONNX Runtime 加载 ----------
sess=ort.InferenceSession(ONNX,providers=["CPUExecutionProvider"])
def ort推理(x): return sess.run(["logits"],{"mel":x.astype(np.float32)})[0]
def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

# ---------- 4. 数值一致性校验（随机输入）----------
xd=dummy.numpy()
with torch.no_grad(): pt_out=em(dummy).numpy()
ort_out=ort推理(xd)
差=np.abs(pt_out-ort_out).max()
print(f"\n【数值校验】PyTorch vs ONNX 最大误差 = {差:.2e}（<1e-4 即视为完全一致）")
print("结论：", "通过，导出后模型算的分一模一样" if np.allclose(pt_out,ort_out,atol=1e-4,rtol=1e-3) else "不一致，需排查")

# ---------- 5. 真实音频：两个引擎各跑一个鱼声/一个水声 ----------
rows=list(csv.DictReader(open(os.path.join(ROOT,"数据划分_manifest.csv"),encoding="utf-8-sig")))
def 取(类别):
    r=next(r for r in rows if r["类别"]==类别)
    y=读波形(os.path.join(ROOT,类别,r["文件名"])); M=(转mel(y)-mean)/std
    return r["文件名"],M[None].astype(np.float32)
for 类别 in ["fish","water"]:
    name,x=取(类别)
    with torch.no_grad(): p_pt=softmax(em(torch.from_numpy(x)).numpy())[0]
    p_ort=softmax(ort推理(x))[0]
    print(f"\n样本 {类别}：{name[:42]}")
    print("  PyTorch 概率：", " ".join(f"{l}={v:.2f}" for l,v in zip(LABELS,p_pt)))
    print("  ONNX    概率：", " ".join(f"{l}={v:.2f}" for l,v in zip(LABELS,p_ort)))
    print(f"  判定：PyTorch→{LABELS[p_pt.argmax()]}  ONNX→{LABELS[p_ort.argmax()]}")
