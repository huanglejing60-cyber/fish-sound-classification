# 配置与参数说明

所有超参数集中在 `src/fishsound/config.py`，也可用环境变量覆盖数据路径。

## 数据路径（环境变量）

| 环境变量 | 含义 | 默认 |
| --- | --- | --- |
| `FISH_DATA_ROOT` | 训练集目录（含 4 类子目录与 manifest） | `E:\大黄鱼分析\dataset_train_v1` |
| `FISH_MODEL_DIR` | 模型权重目录（.pt / .onnx） | `E:\大黄鱼分析\models` |
| `FISH_REC_DIR` | 全天长录音目录 | 成鱼 30054 目录 |

## 音频 / log-mel

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `SAMPLE_RATE` | 32000 | 原始 96k 统一重采样到 32k |
| `N_FFT` | 1024 | FFT 窗长 |
| `HOP_LENGTH` | 320 | 帧移（300 帧 ≈ 3 秒） |
| `N_MELS` | 64 | mel 频带数 |
| `FMIN/FMAX` | 50 / 14000 Hz | 频率范围（覆盖鱼声 0.7–0.9 kHz） |
| `WIN_SEC/SEG_SEC` | 3.0 / 3.0 | 滑窗窗口 / 步长（无重叠） |

## 检测阈值（精度优先，接大数据后需重定）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `FISH_THRESH` | 0.50 | 单窗报鱼阈值 |
| `SEG_THRESH` | 0.30 | 平滑后的成片阈值 |
| `SMOOTH_WIN` | 20 | 1 分钟居中滑动平均窗口 |

## 训练

| 参数 | 值 |
| --- | --- |
| batch / 梯度累积 | 4 / 2（等效 8） |
| 优化器 | AdamW，lr=1e-4，weight_decay=1e-4 |
| 损失 | 类别加权交叉熵 |
| 微调范围 | conv_block5/6 + fc1 + 分类头（浅层冻结，BN 保持 eval） |
| 精度 | AMP 混合精度 + GradScaler |

## 量化

`QUANT_SENSITIVE_NODES`：首层卷积 + 分类头保留 fp32（混合精度），其余 INT8。
