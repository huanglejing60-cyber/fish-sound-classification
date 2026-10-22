# -*- coding: utf-8 -*-
"""模型定义：CNN14（AudioSet 预训练骨干网络）。

CNN14 是 PANNs 系列的音频事件分类网络，输入 log-mel (N,64,300)，
6 个卷积块（1->64->128->256->512->1024->2048）+ 全连接头。
本项目用 527 类 AudioSet 权重做"耳朵"，只替换最后 4 类分类头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def init_layer(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + BN + ReLU + 2x2 平均池化。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))


class Cnn14(nn.Module):
    def __init__(self, classes=4, n_mels=C.N_MELS):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        # (N,64,300) -> (N,1,64,300) -> 输入 BN -> 转回 (N,1,F,T)
        x = x.unsqueeze(1).permute(0, 2, 3, 1)
        x = self.bn0(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        # 时间/频率聚合
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x = x1 + torch.mean(x, dim=2)
        x = F.dropout(x, 0.5, self.training)
        x = F.relu_(self.fc1(x))
        emb = F.dropout(x, 0.5, self.training)
        return self.fc_audioset(emb), emb


class InferenceWrapper(nn.Module):
    """部署外壳：上线只输出 logits，不输出中间 embedding。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def build_pretrained_cnn14(pretrained_path=C.PRETRAIN_SAFETENSORS,
                           classes=len(C.LABELS)):
    """加载 527 类 AudioSet 预训练权重，再换成本任务的分类头。"""
    from safetensors.torch import load_file

    raw = load_file(pretrained_path)
    state = {}
    for k, v in raw.items():
        if not k.startswith("backbone."):
            continue
        nk = k[len("backbone."):]
        if nk.startswith("spectrogram_extractor") or nk.startswith("logmel_extractor"):
            continue
        state[nk] = v
    model = Cnn14(527)
    model.load_state_dict(state, strict=True)
    model.fc_audioset = nn.Linear(2048, classes)
    init_layer(model.fc_audioset)
    return model


# ==========================================================================
# 轻量模型（用于端侧模型选型对比）：均为"从零训练"，输入 (N,64,300) log-mel，
# 输出 (logits, emb)，与 Cnn14 的接口完全一致，方便统一训练/评估/导出。
# 核心算子是深度可分离卷积（Depthwise-Separable）：把标准 3x3 卷积拆成
# 「逐通道 3x3 卷积（depthwise）+ 1x1 跨通道融合（pointwise）」，
# 计算量/参数量可降到原来的约 1/8～1/9，是手机、MCU 等端侧的标配。
# ==========================================================================

class DepthwiseSeparable(nn.Module):
    """一个深度可分离卷积块：dw 3x3（分组卷积）→ BN/ReLU → pw 1x1 → BN/ReLU。"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        init_layer(self.dw)
        init_layer(self.pw)
        init_bn(self.bn_dw)
        init_bn(self.bn_pw)

    def forward(self, x):
        x = F.relu_(self.bn_dw(self.dw(x)))
        return F.relu_(self.bn_pw(self.pw(x)))


class DsCnn(nn.Module):
    """DS-CNN：ARM 关键词唤醒（KWS）经典的窄而深小网络，极致轻量（约 0.14M）。

    结构：1 个标准卷积做通道投影 → 7 个深度可分离块（通道 32→256）→ 全局
    平均池化 → 4 类全连接。频率/时间方向每两段做一次 2x2 降采样。
    """

    def __init__(self, classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        plan = [(32, 32, 1), (32, 64, 2),
                (64, 64, 1), (64, 128, 2),
                (128, 128, 1), (128, 256, 2),
                (256, 256, 1)]
        self.blocks = nn.Sequential(
            *[DepthwiseSeparable(a, b, s) for a, b, s in plan])
        self.fc = nn.Linear(256, classes)
        init_layer(self.fc)

    def forward(self, x):
        x = x.unsqueeze(1)            # (N,1,64,300)
        x = self.stem(x)
        x = self.blocks(x)
        emb = torch.mean(x, dim=(2, 3))   # 全局平均池化
        return self.fc(emb), emb


class MobileNetAudio(nn.Module):
    """MobileNet 风格：比 DS-CNN 更宽（通道到 1024）、容量更大（约 1.5M）。

    结构沿用 MobileNet-V1 的「标准 stem + 一串深度可分离块」，并在末尾加
    一个 1024→256 的隐藏全连接，兼顾表达能力与轻量。
    """

    def __init__(self, classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        plan = [(32, 64, 1), (64, 128, 2),
                (128, 256, 2), (256, 512, 2),
                (512, 512, 1), (512, 512, 1),
                (512, 1024, 2)]
        self.blocks = nn.Sequential(
            *[DepthwiseSeparable(a, b, s) for a, b, s in plan])
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, classes)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3))
        x = F.dropout(x, 0.3, self.training)
        emb = F.relu_(self.fc1(x))
        emb = F.dropout(emb, 0.3, self.training)
        return self.fc2(emb), emb


def count_params(model):
    """可训练参数总数（单位：个）。"""
    return sum(p.numel() for p in model.parameters())


def build_lightweight(name, classes=len(C.LABELS)):
    """按名字构建轻量模型：'dscnn' 或 'mobilenet'。"""
    name = name.lower()
    if name in ("dscnn", "ds-cnn"):
        return DsCnn(classes)
    if name in ("mobilenet", "mobile"):
        return MobileNetAudio(classes)
    raise ValueError(f"未知轻量模型：{name}（可选 dscnn / mobilenet）")
