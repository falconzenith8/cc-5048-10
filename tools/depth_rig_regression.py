#!/usr/bin/env python3
"""深度测试相机回归套件入口（纯标准库）。

用法：python3 tools/depth_rig_regression.py
退出码 0 表示全部断言通过。
"""
from __future__ import annotations

import os
import sys

# 让 depth_rig 作为独立顶层包可导入，绕过主项目 numpy 依赖
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "occlude"))

from depth_rig.regression import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
