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
