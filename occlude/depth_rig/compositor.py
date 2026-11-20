"""合成内核：传感器帧 -> 最终显示用合成缓冲。

合成器只允许读取传感器对外暴露的观测（颜色、延后深度、有效位、区域
标签），不得读取传感器保留的 true_depth 真值。DisplayBuffer 就是最终
显示缓冲，回归断言直接逐像素读取它，而不是比对中间量。

状态标签（逐像素，写入显示缓冲）：
- OK_VALID          有效深度，按声明规则合成
- MISSING_INVALID   有效位=False：缺测，颜色保留颜色相机画面并标记
- EDGE_UNCERTAIN    薄板边缘不确定带：单独标记，不做武断遮挡判定

四种合成器由 CompositorConfig 选择，其中三种是被回归套件注入的故障：
- golden（全 False）：遵守声明规则，缺测不当遮挡、不做孔填充、信任时延；
- temporal_mismatch：忽略深度流的一帧时延，用当前帧几何深度替代传感器
  深度（典型的时间错配 bug）；
- hole_fill：在孔内可见区用薄板深度回填，制造「孔内补深度」；
- missing_as_occluder：把所有有效位=False 的像素一律当作最近遮挡物
  （深度记为薄板深度、颜色替换为板色），用于检验缺测规则。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from depth_rig.geometry import EDGE_UNCERTAIN, HOLE_VISIBLE, OCCLUDED, Grid
from depth_rig.scenario import Scenario
from depth_rig.sensor import FrameObservation, raycast_surface

OK_VALID = "ok_valid"
MISSING_INVALID = "missing_invalid"
EDGE_STATUS = "edge_uncertain"

STATUS_CODES = {
    OK_VALID: 0,
    MISSING_INVALID: 1,
    EDGE_STATUS: 2,
}


@dataclass(frozen=True)
class CompositorConfig:
    temporal_mismatch: bool = False
    hole_fill: bool = False
    missing_as_occluder: bool = False


@dataclass
class DisplayBuffer:
    frame_index: int
    timestamp_s: float
    width_px: int
    height_px: int
    color: list[list[tuple[int, int, int]]]
    depth_mm: list[list[float]]
    status: list[list[str]]
    config: CompositorConfig
    stats: dict = field(default_factory=dict)

    def status_mask(self, label: str) -> list[list[bool]]:
        return [[cell == label for cell in row] for row in self.status]


def _current_geometry_depth(scenario: Scenario, frame: FrameObservation,
                            r: int, c: int) -> float:
    """时间错配故障专用：绕过传感器时延，直接按当前帧板位姿求深度。"""
    x, y = scenario.grid.pixel_to_ray(r, c)
    label = frame.regions[r][c]
    if label in (OCCLUDED, EDGE_UNCERTAIN):
        return scenario.plate.z_plate_mm
    hit = raycast_surface(scenario.surfaces, x, y)
    return hit.z if hit is not None else 0.0


def composite_frame(scenario: Scenario, frame: FrameObservation,
                    config: CompositorConfig | None = None) -> DisplayBuffer:
    cfg = config or CompositorConfig()
    grid: Grid = scenario.grid
    h, w = grid.height_px, grid.width_px

    out_color = [row.copy() for row in frame.color]
    out_depth = [[0.0] * w for _ in range(h)]
    status = [[OK_VALID] * w for _ in range(h)]

    n_missing = n_edge = n_hole_fill = n_temporal = 0
    for r in range(h):
        for c in range(w):
            label = frame.regions[r][c]

            # 边缘不确定带优先级最高：即使该像素同时落在声明无效区，
            # 也先按边缘带处理，保证三类区域标签互斥、不重复计数。
            if label == EDGE_UNCERTAIN:
                out_depth[r][c] = frame.depth_mm[r][c]
                status[r][c] = EDGE_STATUS
                n_edge += 1
                continue

            if not frame.valid[r][c]:
                if cfg.missing_as_occluder:
                    # 故障：缺测一律当成最近遮挡物
                    out_color[r][c] = scenario.plate.color
                    out_depth[r][c] = scenario.plate.z_plate_mm
                    status[r][c] = OK_VALID
                else:
                    # 声明规则：保留颜色相机画面，深度缺测为 0 并明确标记
                    out_depth[r][c] = 0.0
                    status[r][c] = MISSING_INVALID
                    n_missing += 1
                continue

            depth = frame.depth_mm[r][c]
            if cfg.temporal_mismatch:
                fresh = _current_geometry_depth(scenario, frame, r, c)
                if abs(fresh - depth) > 1e-9:
                    n_temporal += 1
                depth = fresh

            if label == HOLE_VISIBLE and cfg.hole_fill:
                depth = scenario.plate.z_plate_mm
                n_hole_fill += 1

            out_depth[r][c] = depth

    return DisplayBuffer(
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        width_px=w, height_px=h,
        color=out_color, depth_mm=out_depth, status=status,
        config=cfg,
        stats={
            "missing_invalid_px": n_missing,
            "edge_uncertain_px": n_edge,
            "hole_fill_px": n_hole_fill,
            "temporal_overwritten_px": n_temporal,
            "missing_as_occluder": cfg.missing_as_occluder,
        },
    )
