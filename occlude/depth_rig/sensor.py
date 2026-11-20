"""深度测试相机传感器：真实颜色 + 延后一帧的深度 + 有效位。

观测三件套逐像素生成：

- color：当前帧真实颜色（颜色相机无时延），板实体给板色，孔内/板外
  给射线命中的最近展品表面颜色；无效像素仍保留颜色相机拍到的颜色。
- depth_mm：深度流固定延后一帧——第 t 帧发出的深度由第 t-1 帧的板位姿
  几何产生（第 0 帧无前序，只能发当前几何）。展品表面静止，运动的只有
  薄板，因此时延只在板轮廓/孔轮廓扫过的像素上制造错配。
- valid：传感器有效位。被声明为无效区的像素 valid=False 且 depth=0；
  其余像素（含边缘不确定带）valid=True。0 这个数值本身不代表任何含义，
  一切以有效位为准。

unit_error_scale > 1 时注入「单位错误」：传感器把远处表面的深度按比例
放大（模拟毫米/英寸一类单位换算 bug），近处与板深度不变，使故障区域
被限制在远表面可见像素上，与时间错配、孔内补深度的区域天然分离。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from depth_rig.geometry import (
    EDGE_UNCERTAIN,
    OCCLUDED,
    Grid,
    Rect,
)
from depth_rig.scenario import Scenario

# 超过该深度的表面才会被单位错误放大（毫米）
UNIT_ERROR_MIN_DEPTH_MM = 1200.0
BACKGROUND_COLOR = (24, 26, 32)


@dataclass(frozen=True)
class SensorConfig:
    depth_lag_frames: int = 1
    unit_error_scale: float = 1.0
    unit_error_min_depth_mm: float = UNIT_ERROR_MIN_DEPTH_MM


@dataclass
class FrameObservation:
    frame_index: int
    timestamp_s: float
    plate_center_x_mm: float
    regions: list[list[str]]
    color: list[list[tuple[int, int, int]]]
    depth_mm: list[list[float]]
    valid: list[list[bool]]
    # 当前帧真实几何深度（无时延），仅供真值核对，合成器不得读取
    true_depth_mm: list[list[float]] = field(repr=False, default=None)


def raycast_surface(surfaces: tuple[Rect, ...], x: float, y: float) -> Rect | None:
    hit: Rect | None = None
    for s in surfaces:
        if s.x0 <= x <= s.x1 and s.y0 <= y <= s.y1:
            if hit is None or s.z < hit.z:
                hit = s
    return hit


def _in_invalid_patch(patches: tuple[Rect, ...], x: float, y: float) -> Rect | None:
    for p in patches:
        if p.x0 <= x <= p.x1 and p.y0 <= y <= p.y1:
            return p
    return None


def _geometry_at(frame_index: int, scenario: Scenario) -> dict:
    """某一帧板位姿下的逐像素几何（颜色、真实深度、区域）。"""
    grid: Grid = scenario.grid
    center_x = scenario.plate_center_x(frame_index)
    regions = grid.region_map(scenario.plate, center_x)
    color, true_depth = [], []
    for r in range(grid.height_px):
        c_row, d_row = [], []
        for c in range(grid.width_px):
            x, y = grid.pixel_to_ray(r, c)
            label = regions[r][c]
            if label in (OCCLUDED, EDGE_UNCERTAIN):
                c_row.append(scenario.plate.color)
                d_row.append(scenario.plate.z_plate_mm)
                continue
            hit = raycast_surface(scenario.surfaces, x, y)
            if hit is None:
                c_row.append(BACKGROUND_COLOR)
                d_row.append(float("inf"))
            else:
                c_row.append(hit.color)
                d_row.append(hit.z)
        color.append(c_row)
        true_depth.append(d_row)
    return {
        "center_x": center_x,
        "regions": regions,
        "color": color,
        "true_depth": true_depth,
    }


def simulate_camera(scenario: Scenario, config: SensorConfig | None = None
                    ) -> list[FrameObservation]:
    cfg = config or SensorConfig()
    grid = scenario.grid
    h, w = grid.height_px, grid.width_px

    geom = [_geometry_at(t, scenario) for t in range(scenario.n_frames)]

    # 有效位只取决于被声明的无效区，逐帧不变（板扫过也不改写传感器标志）
    valid_flag: list[list[bool]] = []
    for r in range(h):
        row = []
        for c in range(w):
            x, y = grid.pixel_to_ray(r, c)
            row.append(_in_invalid_patch(scenario.invalid_patches, x, y) is None)
        valid_flag.append(row)

    frames: list[FrameObservation] = []
    for t in range(scenario.n_frames):
        src = t - cfg.depth_lag_frames
        if src < 0:
            src = t  # 第 0 帧无前序时间戳，深度只能取当前几何
        depth_src = geom[src]["true_depth"]
        regions_src = geom[src]["regions"]
        regions_prev = geom[max(0, src - 1)]["regions"]
        regions_now_t = geom[t]["regions"]
        depth: list[list[float]] = []
        for r in range(h):
            row = []
            for c in range(w):
                if not valid_flag[r][c]:
                    row.append(0.0)
                    continue
                d = depth_src[r][c]
                # 单位错误只作用于板外自由射线上、且来源帧前后两侧区域
                # 标签都稳定的远表面。任一侧标签在变化即属于时间错配扫动
                # 条带，留给 temporal_mismatch，保证两类失败区互不重叠。
                is_unit_target = (
                    cfg.unit_error_scale != 1.0
                    and regions_src[r][c] == "free"
                    and regions_prev[r][c] == regions_src[r][c]
                    and regions_now_t[r][c] == regions_src[r][c]
                    and d >= cfg.unit_error_min_depth_mm)
                if d == float("inf"):
                    row.append(0.0)  # 无回波：远墙覆盖全视场，此分支实际不触发
                elif is_unit_target:
                    row.append(d * cfg.unit_error_scale)
                else:
                    row.append(d)
            depth.append(row)
        frames.append(FrameObservation(
            frame_index=t,
            timestamp_s=scenario.timestamp_s(t),
            plate_center_x_mm=geom[t]["center_x"],
            regions=geom[t]["regions"],
            color=geom[t]["color"],
            depth_mm=depth,
            valid=valid_flag,
            true_depth_mm=geom[t]["true_depth"],
        ))
    return frames
