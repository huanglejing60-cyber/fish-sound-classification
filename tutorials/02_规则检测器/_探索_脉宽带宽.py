# -*- coding: utf-8 -*-
# 探索3：从成组咕咕声里取单个脉冲（真正的鱼声），测脉宽/中心频率/带宽
import numpy as np, librosa, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]; plt.rcParams["axes.unicode_minus"]=False
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
sr=16000
y,_=librosa.load(wav,sr=sr,mono=True)
w=sr*10; n=len(y)//w
best,bk=-1,0
for i in range(n):
    s=y[i*w:(i+1)*w]; r=np.sqrt(np.mean(s**2))
    if r>best: best,bk=r,i
y10=y[bk*w:(bk+1)*w]; base=bk*10
env=np.abs(hilbert(y10)); k=int(0.0003*sr); env=np.convolve(env,np.ones(k)/k,mode="same")
med=np.median(env); height=med+0.06*(env.max()-med)
peaks,_=find_peaks(env,distance=int(0.01*sr),height=height,prominence=0.02)
pt=peaks/sr; gaps=np.diff(pt)*1000
# 选间隔80-150ms的成组脉冲
组成员=[]
for j in range(1,len(peaks)-1):
    if 80<=gaps[j-1]<=150 and 80<=gaps[j]<=150:
        组成员.append((peaks[j],base+pt[j]))
print("成组中脉冲数:",len(组成员))
sr96=96000
w50s=[];fcs=[];bws=[]
for p16,abst in 组成员[:8]:
    seg96,_=librosa.load(wav,sr=sr96,mono=True,offset=abst-0.01,duration=0.02)
    e96=np.abs(hilbert(seg96)); kk=max(1,int(0.0002*sr96)); e96=np.convolve(e96,np.ones(kk)/kk,mode="same")
    pk=np.argmax(e96)
    above=np.where(e96>=e96[pk]*0.5)[0]; w50=(above[-1]-above[0])/sr96*1000
    A=np.abs(np.fft.rfft(seg96*np.hanning(len(seg96)),n=65536)); f=np.fft.rfftfreq(65536,1/sr96)
    m=(f>=400)&(f<=1600); f,a=f[m],A[m]; f0=f[np.argmax(a)]
    hi=np.where(a>=a.max()/np.sqrt(2))[0]; bw=(f[hi[-1]]-f[hi[0]]) if len(hi) else 0
    w50s.append(w50);fcs.append(f0);bws.append(bw)
print(f"成组鱼声脉冲: 50%脉宽中位={np.median(w50s):.2f}ms  中心频率中位={np.median(fcs):.0f}Hz  -3dB带宽中位={np.median(bws):.0f}Hz")
print("各脉冲中心频率:",[int(x) for x in fcs])
print("各脉冲带宽:",[int(x) for x in bws])
