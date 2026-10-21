# -*- coding: utf-8 -*-
"""
=====================================================================
第 19 课 b：部署到底快多少、小多少？—— PyTorch vs ONNX Runtime
---------------------------------------------------------------------
测三种"跑法"的延迟（处理一个3秒窗口要多少毫秒）：
  PyTorch CPU / ONNX Runtime CPU / PyTorch GPU
还测 batch=1（实时用，一来一个）和 batch=16（批量扫全天用）。
延迟取中位数（多次计时，先预热，第一次编译不算）。
=====================================================================
"""
import os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
MODEL_DIR=r"E:\大黄鱼分析\models"; V2=os.path.join(MODEL_DIR,"cnn14_finetune_v2_clean.pt"); ONNX=os.path.join(MODEL_DIR,"cnn14_v2.onnx")
N_MELS=64; N_FRAMES=300
# （模型结构与19a一致，简装）
def init_layer(l):
    if isinstance(l,(nn.Linear,nn.Conv2d)):
        nn.init.xavier_uniform_(l.weight)
        if l.bias is not None: l.bias.data.fill_(0.)
def init_bn(bn): bn.bias.data.fill_(0.); bn.weight.data.fill_(1.)
class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__(); self.conv1=nn.Conv2d(ci,co,3,padding=1,bias=False); self.conv2=nn.Conv2d(co,co,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(co); self.bn2=nn.BatchNorm2d(co); init_layer(self.conv1); init_layer(self.conv2); init_bn(self.bn1); init_bn(self.bn2)
    def forward(self,x):
        x=F.relu_(self.bn1(self.conv1(x))); x=F.relu_(self.bn2(self.conv2(x))); return F.avg_pool2d(x,2,2)
class Cnn14(nn.Module):
    def __init__(self,classes=4):
        super().__init__(); self.bn0=nn.BatchNorm2d(N_MELS)
        self.conv_block1=ConvBlock(1,64); self.conv_block2=ConvBlock(64,128); self.conv_block3=ConvBlock(128,256)
        self.conv_block4=ConvBlock(256,512); self.conv_block5=ConvBlock(512,1024); self.conv_block6=ConvBlock(1024,2048)
        self.fc1=nn.Linear(2048,2048); self.fc_audioset=nn.Linear(2048,classes); init_bn(self.bn0); init_layer(self.fc1); init_layer(self.fc_audioset)
    def forward(self,x):
        x=x.unsqueeze(1).permute(0,2,3,1); x=self.bn0(x); x=x.permute(0,3,2,1)
        x=self.conv_block1(x); x=self.conv_block2(x); x=self.conv_block3(x); x=self.conv_block4(x)
        x=self.conv_block5(x); x=self.conv_block6(x)
        x=torch.mean(x,dim=3); x1,_=torch.max(x,dim=2); x=x1+torch.mean(x,dim=2)
        x=F.dropout(x,0.5,self.training); x=F.relu_(self.fc1(x)); emb=F.dropout(x,0.5,self.training)
        return self.fc_audioset(emb),emb

ck=torch.load(V2,map_location="cpu",weights_only=False)
m_cpu=Cnn14(4); m_cpu.load_state_dict(ck["state_dict"]); m_cpu.eval()
sess=ort.InferenceSession(ONNX,providers=["CPUExecutionProvider"])
有GPU=torch.cuda.is_available()
if 有GPU:
    m_gpu=Cnn14(4); m_gpu.load_state_dict(ck["state_dict"]); m_gpu.eval().cuda()

def pt_cpu(x):
    with torch.no_grad(): m_cpu(torch.from_numpy(x))
def pt_gpu(x):
    with torch.no_grad(): m_gpu(torch.from_numpy(x).cuda()); torch.cuda.synchronize()
def onnx_cpu(x): sess.run(["logits"],{"mel":x.astype(np.float32)})
def 测(fn,x,n=80,预热=8):
    for _ in range(预热): fn(x)
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(x); ts.append((time.perf_counter()-t0)*1000)
    return np.median(ts), np.percentile(ts,95)

结果={}
for bs in [1,16]:
    x=np.random.randn(bs,N_MELS,N_FRAMES).astype(np.float32)
    结果[f"batch={bs}"]={
        "PyTorch CPU":测(pt_cpu,x),
        "ONNX Runtime CPU":测(onnx_cpu,x),
        **({"PyTorch GPU":测(pt_gpu,x)} if 有GPU else {})}
    print(f"\n===== batch={bs}（一次处理{bs}个3秒窗口）延迟中位数/p95（毫秒）=====")
    for k,(med,p95) in 结果[f"batch={bs}"].items(): print(f"  {k:18s} {med:7.2f} / {p95:7.2f}")

# 实时性 & 全天扫描耗时（按batch=1的ORT）
ort单窗=结果["batch=1"]["ONNX Runtime CPU"][0]
print(f"\n【实时性】一个3秒窗口，ONNX CPU 只需 {ort单窗:.1f} ms，比实时快 {3000/ort单窗:.0f} 倍")
print(f"【全天48个30分钟文件=28800个窗口】ONNX CPU 纯推理约 {28800*ort单窗/1000/60:.1f} 分钟（且可批量更快）")

# 体积
sz_pt=os.path.getsize(V2)/1e6; sz_onnx=os.path.getsize(ONNX)/1e6
print(f"\n【模型体积】训练产物 .pt = {sz_pt:.0f} MB；导出 .onnx(fp32) = {sz_onnx:.0f} MB")

# 画图
fig,ax=plt.subplots(1,2,figsize=(13,5))
for i,bs in enumerate(["batch=1","batch=16"]):
    d=结果[bs]; names=list(d.keys()); meds=[d[n][0] for n in names]; p95=[d[n][1] for n in names]
    colors=["#9ca3af","#2563eb"]+["#16a34a" if 有GPU else "#ccc"]
    b=ax[i].bar(names,meds,color=colors[:len(names)],yerr=[np.array(p95)-np.array(meds)],capsize=4)
    for r,v in zip(b,meds): ax[i].text(r.get_x()+r.get_width()/2,v+1,f"{v:.0f}ms",ha="center",fontsize=11)
    ax[i].set_title(f"{bs} 推理延迟（越矮越快）"); ax[i].set_ylabel("延迟(毫秒)"); ax[i].grid(alpha=.3,axis="y")
plt.suptitle("第19课b 部署速度：ONNX Runtime vs PyTorch（中位数，竖线是p95）")
plt.tight_layout(); plt.savefig(r"E:\大黄鱼分析\学习\第19课b_延迟对比.png",dpi=140,bbox_inches="tight")
print("图在 学习/第19课b_延迟对比.png")
