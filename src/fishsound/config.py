# -*- coding: utf-8 -*-
"""全局配置：音频 / log-mel / 滑窗 / 类别 / 数据路径。

所有路径都可以用环境变量覆盖，方便别人 clone 后改成自己的数据位置；
不设置环境变量时默认使用本机的 E 盘路径。
"""
import os

# ---------------- 音频与 log-mel 参数（与训练口径一致） ----------------
SAMPLE_RATE = 32000      # 原始录音 96k，统一重采样到 32k
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 64
FMIN = 50
FMAX = 14000
N_FRAMES = 300           # 300 帧 ≈ 3 秒（300*HOP/SR）
WIN_SEC = 3.0            # 滑窗窗口长度（秒）
SEG_SEC = 3.0            # 滑窗步长（秒），无重叠时等于窗口长度

# ---------------- 类别（fish=0 是核心目标类） ----------------
LABELS = ["fish", "stress", "water", "impact"]
LABELS_CN = {
    "fish": "真鱼",
    "stress": "应激",
    "water": "水流设备",
    "impact": "撞击",
}

# ---------------- 推理 / 检测阈值（精度优先，宁漏勿错） ----------------
FISH_THRESH = 0.50       # 单窗报鱼阈值
SEG_THRESH = 0.30        # 平滑后的成片阈值
SMOOTH_WIN = 20          # 1 分钟居中滑动平均窗口（每窗 3s）

# ---------------- 路径（按需修改或用环境变量覆盖） ----------------
DATA_ROOT = os.environ.get("FISH_DATA_ROOT", r"E:\大黄鱼分析\dataset_train_v1")
MODEL_DIR = os.environ.get("FISH_MODEL_DIR", r"E:\大黄鱼分析\models")
RECORDING_DIR = os.environ.get(
    "FISH_REC_DIR", r"E:\大黄鱼数据\#30054#[020]@20250313_085156"
)
PRETRAIN_SAFETENSORS = os.path.join(MODEL_DIR, "cnn14_hf.safetensors")

# 训练/推理常用文件
MANIFEST = "数据划分_manifest.csv"
CLEAN_MODEL = "cnn14_finetune_v2_clean.pt"
ONNX_MODEL = "cnn14_v2.onnx"
INT8_MIXED_MODEL = "cnn14_v2_int8_mix.onnx"

# 对量化最敏感、需要保留 fp32 的层（第一层卷积 + 最后分类头）。
# 注意：节点名前缀取决于导出时的包装层（InferenceWrapper 里是 /model/）。
# deploy.quantize_mixed_int8 会在量化前校验这些节点真实存在，避免静默失效。
QUANT_SENSITIVE_NODES = [
    "/model/conv_block1/conv1/Conv",
    "/model/conv_block1/conv2/Conv",
    "/model/fc_audioset/Gemm",
]
