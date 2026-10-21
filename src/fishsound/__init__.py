# -*- coding: utf-8 -*-
"""fishsound：大黄鱼声信号分类（音频事件检测 + 迁移学习 + 部署）。

快速入口：
    from fishsound import config, features, models, data, train, infer, deploy
"""
from . import config          # noqa: F401
from . import features        # noqa: F401
from . import models          # noqa: F401
from . import data            # noqa: F401
from . import augment         # noqa: F401
from . import metrics         # noqa: F401
from . import train           # noqa: F401
from . import infer           # noqa: F401
from . import deploy          # noqa: F401

__version__ = "1.0.0"
__all__ = ["config", "features", "models", "data", "augment",
           "metrics", "train", "infer", "deploy"]
