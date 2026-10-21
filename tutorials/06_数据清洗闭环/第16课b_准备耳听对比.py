# -*- coding: utf-8 -*-
"""第16课b：准备"真鱼基准 + 5个各时段代表"供耳听"""
import os, shutil
import numpy as np, librosa, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
ROOT=r"E:\大黄鱼分析\dataset_train_v1"
OUT=os.path.join(ROOT,"_待复核标签")
HEAR=os.path.join(ROOT,"先听这6个")
os.makedirs(HEAR,exist_ok=True)
SR=32000
# (源文件名, 易懂新名, 副标题)
picks=[
 ("v3_C_真产卵1921_真实192511_脉冲41_CV0.68.wav","z00_基准真鱼_19点25.wav","基准：真产卵咕咕"),
 ("f01_034554_val.wav","z01_凌晨3点45.wav","凌晨3:45（非产卵）"),
 ("f05_100420_val.wav","z02_放鱼后10点04.wav","放鱼结束后15分钟"),
 ("f12_164604_train.wav","z03_喂食16点46.wav","下午4:46喂食前后"),
 ("f17_174143_train.wav","z04_人工种子17点41.wav","喂食换水段·原标fish"),
 ("f22_222702_train.wav","z05_夜间22点27.wav","夜间22:27"),
]
fig,axes=plt.subplots(2,3,figsize=(15,7))
for ax,(src,new,sub) in zip(axes.reshape(-1),picks):
    sp=os.path.join(ROOT,"fish",src) if src.startswith("v3") else os.path.join(OUT,src)
    dp=os.path.join(HEAR,new); shutil.copy(sp,dp)
    y,sr=sf.read(dp)
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    M=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=320,n_mels=64,fmin=50,fmax=14000)
    D=librosa.power_to_db(M,ref=1.0,amin=1e-10)
    ax.imshow(D,origin="lower",aspect="auto",cmap="magma",extent=[0,3,50,14000])
    ax.set_title(new.replace(".wav","")+"\n"+sub,fontsize=10,
                 color="#15803d" if new.startswith("z00") else "#b91c1c")
    ax.set_xlabel("秒");ax.set_ylabel("Hz")
plt.suptitle("先听 z00 记住真鱼咕咕声，再听 z01–z05 判断像不像鱼",fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(HEAR,"耳听对比.png"),dpi=130,bbox_inches="tight")
print("已生成到：",HEAR)
for f in sorted(os.listdir(HEAR)): print(" ",f)
