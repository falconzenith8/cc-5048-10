"""pytest 收集入口：复用 depth_rig.regression 的全部断言。

环境装有 pytest 时可直接 `pytest tools/test_depth_rig.py`；本文件仍只
依赖标准库，导入 depth_rig 时自行引导 sys.path，不触发主项目 numpy 依赖。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "occlude"))

from depth_rig.regression import run_suite  # noqa: E402


def test_depth_rig_regression_all_pass():
    results = run_suite()
    failed = [f"{r.name}: {r.detail}" for r in results if not r.passed]
    assert not failed, "深度测试相机回归失败：\n" + "\n".join(failed)
