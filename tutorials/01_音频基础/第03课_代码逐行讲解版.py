# -*- coding: utf-8 -*-
# =====================================================================
# 第03课 · 代码逐行讲解版（Python新手对照用）
# 绿色【】里是 Python 语法知识点；看懂这一份，你就能看懂前面所有脚本
# =====================================================================

# ---------- 一、import：把"工具箱"搬进来 ----------
# 【库】就是别人写好的一堆功能；【import 库名】= 把这个工具箱拿来用
# 【as 别名】= 给工具箱起个短名，省得每次打全名，np/plt 是全行业通用简称
import numpy as np                    # numpy：专门处理数字数组（音频本质就是数组）
import librosa                        # librosa：音频分析工具箱
import matplotlib
matplotlib.use("Agg")                 # 让图直接存成文件，不弹窗（必须在pyplot之前设置）
import matplotlib.pyplot as plt       # pyplot 是画图模块，起别名 plt
from scipy.signal import find_peaks, hilbert   # 【from...import】只从scipy里拿这两个工具
# 【点号 . 】表示"里面的"：scipy.signal 意思是 scipy 这个库里面的 signal 模块

# 让中文不乱码（固定两行，背下来即可）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 二、变量：给数据贴个名字 ----------
# 【变量】= 名字 + 等号 + 数据，等号是"赋值"（把右边放进左边），不是数学等号
wav = r"E:\大黄鱼数据\#30054#[020]@20250313_085156\#30054#[020]@20250313_185156@MID@96Ksps.wav"
# 上面是字符串（文本）；前面的 r 表示原始字符串，让路径里的反斜杠 \ 不当特殊符号
# 注意：路径必须是真实存在的完整文件，不能用 ... 省略，否则程序找不到文件
sr = 16000                            # 整数：目标采样率
y, _ = librosa.load(wav, sr=sr, mono=True)
# 【函数】= 别人封装好的一段动作，写法是 函数名(参数)，返回结果
# 【参数】括号里传给函数的东西：wav 是位置参数；sr=sr 是"关键字参数(名字=值)"
# 【返回值】load会返回两个东西：(音频数据, 采样率)，用逗号同时接收
# 【_】下划线是约定俗成的"我不要这个返回值"，这里只要音频y

# ---------- 三、numpy数组：音频的真身 ----------
# y 是一个 numpy 数组，可以理解成"一串有序的数字"，共 16000*1800 个
窗长 = sr * 10                        # 变量可以用中文（Python3支持），=10秒的采样点数
n窗 = len(y) // 窗长                  # len(y)=数组长度；// 是"整除"（除完只留整数）
best, bk = -1, 0                      # 同时给两个变量赋值：best=-1（记录最大响度），bk=0（记录第几段）

# ---------- 四、for 循环：重复做同一件事 ----------
# 【for x in range(n):】让 x 依次取 0,1,2,...,n-1；冒号结尾
for i in range(n窗):
    # 【缩进】Python用4个空格表示"这一段属于for"，缩进没了循环就结束
    seg = y[i*窗长:(i+1)*窗长]        # 【切片】取数组第 a 到 b 个，写法 y[a:b]（含a不含b）
    r = np.sqrt(np.mean(seg**2))      # 这一句从里向外算：
    #   seg**2   → 【**是乘方】每个元素平方（**2=平方），得到能量
    #   np.mean()→ 所有元素求平均
    #   np.sqrt()→ 开平方；连起来就是 RMS 均方根响度
    if r > best:                      # 【if判断】如果 r 比 best 大（>是比较，结果True/False）
        best, bk = r, i               # 满足才执行这行：记下更响的响度和段号
# 【小结】这个循环：把整段切成10秒一段，找出最响的那一段

# ---------- 五、取出最响10秒，造时间轴 ----------
y10 = y[bk*窗长:(bk+1)*窗长]          # 再用切片，把最响那10秒取出来
t = np.arange(len(y10)) / sr          # np.arange(n)生成 0,1,2,...,n-1；除以sr → 每个点对应的秒数

# ---------- 六、包络（信号处理）----------
analytic = hilbert(y10)               # 调用hilbert工具，得到"解析信号"
env = np.abs(analytic)                # np.abs 对每个元素取绝对值（复数就是取模）=瞬时幅度
k = int(0.005 * sr)                   # int()把小数转成整数（5ms对应的采样点数）
env = np.convolve(env, np.ones(k)/k, mode="same")
# np.ones(k)生成k个1；除以k → 长度为k的"滑动平均"小窗
# np.convolve卷积 = 让这条曲线逐点平均、变平滑（5ms滑动平均）

# ---------- 七、用变量做"阈值"（检测器的核心）----------
med = np.median(env)                  # np.median 中位数（底噪水平，比平均值更抗突发干扰）
height = med + 0.06 * (env.max() - med)
# env.max()是数组的【方法】：让env自己算最大值。等价于 np.max(env)
# 【函数 vs 方法】np.max(env)把数据传进去；env.max()让对象自己算，效果一样

# ---------- 八、find_peaks 找每一声"咕" ----------
peaks, _ = find_peaks(env, distance=int(0.02*sr), height=height, prominence=0.02)
# 多个参数用逗号隔开；distance/height/prominence 都是在规定"什么样才算一个峰"
peak_t = peaks / sr                   # 峰的"采样点序号"除以采样率 → 峰的"秒数"
gaps_ms = np.diff(peak_t) * 1000      # np.diff：相邻两项做差（后-前）=间隔；×1000秒转毫秒

# ---------- 九、布尔数组做筛选 ----------
# 【比较数组】对数组做比较，会得到一串 True/False；再用来"挑"元素
组内 = ((gaps_ms >= 80) & (gaps_ms <= 150)).sum()
#  >= 大于等于，<= 小于等于；& 是数组里的"而且"（普通代码里用 and）
#  ...&... 得到True/False数组；.sum() 把True当1数出来 → 满足条件的个数
停顿 = (gaps_ms > 300).sum()          # 统计间隔大于300ms的个数

# ---------- 十、print 输出 + f-string 拼字符串 ----------
print("=" * 46)                       # 字符串 * 数字 = 重复46次"="，打印分割线
print(f"这10秒里共检测到 {len(peaks)} 下鱼声脉冲")
# 【f-string】字符串前加 f，{} 里可以放变量或表达式，会自动算出结果填进去
print(f"相邻脉冲间隔中位数 = {np.median(gaps_ms):.0f} ms")
# 【:.0f】是格式：保留0位小数（即显示成整数）；.1f就是保留1位
print(f"咕咕紧挨着的双/三脉冲：{组内} 个，发声之间停顿：{停顿} 个")
print(f"平均间隔 = {np.mean(gaps_ms):.0f} ms")

# ---------- 十一、画图（固定套路，照抄即可）----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))  # 建2行1列的画布，返回图和两个坐标轴
ax1.plot(t, y10, color="#94a3b8", lw=0.4)              # ax1.plot 在第一个轴画线
ax1.plot(t, env, color="#d6336c", lw=1.2)
ax1.plot(peak_t, env[peaks], "v", color="red", ms=7)   # "v"=倒三角标记，ms=标记大小
ax1.set_title("标题"); ax1.set_xlabel("时间(秒)"); ax1.set_ylabel("振幅")
ax1.set_xlim(0, 10); ax1.grid(alpha=0.3)               # set_xlim设坐标范围；grid加网格
plt.tight_layout()                                      # 自动调整间距
plt.savefig(r"E:\大黄鱼分析\学习\第03课_脉冲间隔测量.png", dpi=130)  # 保存图片

# =====================================================================
# 看完这份，你应该记住的 Python 高频用法（抄到你笔记本上）：
#  1. import 库 as 别名          2. 变量赋值 =，判断相等 ==
#  3. 数组切片 y[a:b]            4. 元素运算 y**2、np.abs(y)
#  5. for i in range(n) + 缩进    6. if 条件: + 缩进
#  7. 函数：函数名(参数)，可有多个返回值 用逗号接收
#  8. print(f"{变量}")  输出时拼变量；【:.0f】控制小数位
# =====================================================================
