# -*- coding: utf-8 -*-
import os, numpy as np, librosa, soundfile as sf
D=r"E:\大黄鱼分析\dataset_train_v1\_待复核标签"; SR=32000
def load(fn):
    y,sr=sf.read(os.path.join(D,fn))
    if y.ndim>1:y=y.mean(1)
    if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
    return y
def 切(fn,start,end,out):
    y=load(fn); sf.write(os.path.join(D,out),y[int(start*SR):int(end*SR)],SR); print("已切",out)
# f14 规律成组段（0.7-2.7秒）
f14=[x for x in os.listdir(D) if x.startswith("f14") and x.endswith(".wav") and "第" not in x][0]
切(f14,0.7,2.7,"f14_规律成组段.wav")
# f13 前半（成对，0-0.7秒）与后半（杂乱，1.6-2.5秒）
f13=[x for x in os.listdir(D) if x.startswith("f13") and x.endswith(".wav") and "第" not in x][0]
切(f13,0.0,0.7,"f13_前半_像鱼成对.wav")
切(f13,1.6,2.5,"f13_后半_杂乱扑腾.wav")
