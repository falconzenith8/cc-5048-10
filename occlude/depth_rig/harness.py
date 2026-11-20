"""测试线束与回放接口。

RegressionHarness 负责把想定、传感器配置、合成器配置组合成一条完整
流水线，并提供逐像素的失败区对比：以「正确（golden）合成器的最终显示
缓冲」为基准，故障缓冲与它逐像素不同的位置即失败像素。

ReplayInterface 对应需求中的「把单个失败帧及前序时间戳送回合成界面」：
它可以用不同的薄板速度（拖慢薄板）或不同的板平面距离重新合成同一帧，
用于区分故障影响随速度还是随距离变化。回放返回的仍是 DisplayBuffer，
界面与测试都直接读最终显示缓冲。
"""
from __future__ import annotations

from dataclasses import dataclass

from depth_rig.compositor import (
    CompositorConfig,
    DisplayBuffer,
    composite_frame,
)
from depth_rig.scenario import (
    PLATE_SPEED_FAST_MM_PER_FRAME,
    PLATE_SPEED_SLOW_MM_PER_FRAME,
    Scenario,
    build_scenario,
)
from depth_rig.sensor import SensorConfig, simulate_camera

# 四种预设：golden 为正确实现，其余三个为可注入故障
PRESETS = (
    "golden",
    "unit_error",
    "temporal_mismatch",
    "hole_fill",
    "missing_as_occluder",
)

# 单位错误：把远表面深度放大 8%（近/中层与板均低于阈值，不受影响）
UNIT_ERROR_SCALE = 1.08


@dataclass(frozen=True)
class Preset:
    name: str
    sensor: SensorConfig
    compositor: CompositorConfig


def get_preset(name: str) -> Preset:
    if name == "golden":
        return Preset(name, SensorConfig(), CompositorConfig())
    if name == "unit_error":
        return Preset(name, SensorConfig(unit_error_scale=UNIT_ERROR_SCALE),
                      CompositorConfig())
    if name == "temporal_mismatch":
        return Preset(name, SensorConfig(), CompositorConfig(temporal_mismatch=True))
    if name == "hole_fill":
        return Preset(name, SensorConfig(), CompositorConfig(hole_fill=True))
    if name == "missing_as_occluder":
        return Preset(name, SensorConfig(),
                      CompositorConfig(missing_as_occluder=True))
    raise KeyError(f"未知故障预设 {name}，可选 {PRESETS}")


class RegressionHarness:
    """固定想定下运行各预设并对比最终显示缓冲。"""

    def __init__(self, scenario: Scenario | None = None):
        self.scenario = scenario or build_scenario()

    def run_preset(self, preset_name: str) -> list[DisplayBuffer]:
        preset = get_preset(preset_name)
        frames = simulate_camera(self.scenario, preset.sensor)
        return [composite_frame(self.scenario, fr, preset.compositor)
                for fr in frames]

    def failure_mask(self, preset_name: str, frame_index: int
                     ) -> list[list[bool]]:
        golden = self.run_preset("golden")[frame_index]
        faulty = self.run_preset(preset_name)[frame_index]
        return buffer_diff_mask(golden, faulty)

    def failure_region_counts(self, preset_name: str, frame_index: int
                              ) -> dict[str, int]:
        """失败像素按独立射线区域标签分类的计数（不含状态差异）。"""
        mask = self.failure_mask(preset_name, frame_index)
        frames = simulate_camera(self.scenario, get_preset(preset_name).sensor)
        regions = frames[frame_index].regions
        counts: dict[str, int] = {}
        for r, row in enumerate(mask):
            for c, failed in enumerate(row):
                if failed:
                    label = regions[r][c]
                    counts[label] = counts.get(label, 0) + 1
        return counts


def buffer_diff_mask(a: DisplayBuffer, b: DisplayBuffer) -> list[list[bool]]:
    """逐像素失败区：深度或状态不同即失败（颜色差异由前两者派生）。"""
    if (a.width_px, a.height_px) != (b.width_px, b.height_px):
        raise ValueError("显示缓冲尺寸不一致，无法逐像素比较")
    h, w = a.height_px, a.width_px
    return [[a.depth_mm[r][c] != b.depth_mm[r][c]
             or a.status[r][c] != b.status[r][c]
             for c in range(w)] for r in range(h)]


def count_mask(mask: list[list[bool]]) -> int:
    return sum(sum(1 for v in row if v) for row in mask)


def masks_disjoint(a: list[list[bool]], b: list[list[bool]]) -> bool:
    return not any(a[r][c] and b[r][c]
                   for r in range(len(a)) for c in range(len(a[0])))


class ReplayInterface:
    """失败帧回放：可改薄板速度与板平面距离，重新合成并返回显示缓冲。

    每次回放都重新运行整条传感器+合成器流水线，保证前序时间戳（深度
    时延所依赖的 t-1 帧几何）与新的速度/距离自洽，而不是只改单帧。
    """

    def __init__(self, preset_name: str):
        self.preset_name = preset_name

    def replay(self, frame_index: int,
               speed_mm_per_frame: float = PLATE_SPEED_FAST_MM_PER_FRAME,
               plate_z_mm: float | None = None,
               history: int = 1) -> dict:
        if history < 0:
            raise ValueError("history 必须是非负整数")
        preset = get_preset(self.preset_name)
        scenario = build_scenario(
            speed_mm_per_frame=speed_mm_per_frame,
            plate_z_mm=plate_z_mm if plate_z_mm is not None
            else self._default_plate_z())
        frames = simulate_camera(scenario, preset.sensor)
        if not 0 <= frame_index < len(frames):
            raise IndexError(f"帧序号 {frame_index} 超出 [0, {len(frames) - 1}]")
        buffers = [composite_frame(scenario, fr, preset.compositor)
                   for fr in frames]
        lo = max(0, frame_index - history)
        return {
            "frame_index": frame_index,
            "preset": self.preset_name,
            "speed_mm_per_frame": speed_mm_per_frame,
            "plate_z_mm": scenario.plate.z_plate_mm,
            "history_indices": list(range(lo, frame_index + 1)),
            "history_timestamps_s": [scenario.timestamp_s(i)
                                     for i in range(lo, frame_index + 1)],
            "buffers": buffers[lo:frame_index + 1],
            "target_buffer": buffers[frame_index],
        }

    @staticmethod
    def _default_plate_z() -> float:
        return build_scenario().plate.z_plate_mm

    @staticmethod
    def slow_speed() -> float:
        return PLATE_SPEED_SLOW_MM_PER_FRAME

    @staticmethod
    def fast_speed() -> float:
        return PLATE_SPEED_FAST_MM_PER_FRAME
