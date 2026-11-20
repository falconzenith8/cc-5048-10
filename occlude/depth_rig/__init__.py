"""可控深度测试相机（纯标准库，零第三方依赖）。

该子包可独立作为顶层包 depth_rig 加载：运行入口会把本目录的父目录
（occlude/）加入 sys.path，因此不受主项目 numpy 依赖是否安装的影响。
模块内统一使用 `from depth_rig.xxx import ...` 绝对导入。
"""
from depth_rig.compositor import (
    CompositorConfig,
    DisplayBuffer,
    composite_frame,
)
from depth_rig.harness import (
    PRESETS,
    RegressionHarness,
    ReplayInterface,
    get_preset,
)
from depth_rig.regression import run_suite
from depth_rig.scenario import build_scenario
from depth_rig.sensor import SensorConfig, simulate_camera

__all__ = [
    "CompositorConfig",
    "DisplayBuffer",
    "PRESETS",
    "RegressionHarness",
    "ReplayInterface",
    "SensorConfig",
    "build_scenario",
    "composite_frame",
    "get_preset",
    "run_suite",
    "simulate_camera",
]
