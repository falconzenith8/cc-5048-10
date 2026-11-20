"""可控深度测试相机的几何内核（纯标准库）。

相机为沿 +Z 观察的正交相机，单位与主项目一致：毫米，OpenCV 约定
（X 右、Y 下、Z 朝前）。每根像素射线 (px, py) 是一条平行于 Z 轴、
固定 (x, y) 的直线，因此射线与薄板的交点就是该射线在板平面 z=z_plate
处的点，遮挡关系可以逐条射线独立计算，不依赖邻域插值。

薄板横向扫过虚拟展品，每个像素被独立射线划分为三种区域：

- OCCLUDED：射线在板平面的落点位于薄板实体（孔外、边缘带内缩之内）；
- HOLE_VISIBLE：落点位于某个孔的内缩区域，射线穿过孔直达后方展品；
- EDGE_UNCERTAIN：落点距板外轮廓或孔轮廓不超过边缘带半宽，传感器
  无法稳定判定落点属于实体还是孔，单独标为不确定带；
- FREE：薄板未覆盖该射线（板外且远离边缘）。

区域只由当前帧板位姿决定，不读取传感器输出，因此可作为独立真值。
"""
from __future__ import annotations

from dataclasses import dataclass

# 区域标签（逐像素真值，独立于传感器观测）
FREE = "free"
OCCLUDED = "occluded"
HOLE_VISIBLE = "hole_visible"
EDGE_UNCERTAIN = "edge_uncertain"

REGIONS = (FREE, OCCLUDED, HOLE_VISIBLE, EDGE_UNCERTAIN)


@dataclass(frozen=True)
class Rect:
    """与相机平面平行的矩形表面（z 为常量），单位毫米。"""
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    z: float
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Hole:
    """薄板上的矩形通孔。"""
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class Plate:
    """带孔薄板：板平面 z_plate，板宽沿 X，高度沿 Y。"""
    width_mm: float
    height_mm: float
    z_plate_mm: float
    holes: tuple[Hole, ...]
    edge_band_mm: float
    color: tuple[int, int, int] = (150, 150, 160)

    def region_at(self, x: float, y: float, center_x: float) -> str:
        """逐射线区域判定。center_x 为本帧板中心在世界系的 X 坐标。"""
        half_w, half_h = self.width_mm / 2.0, self.height_mm / 2.0
        left, right = center_x - half_w, center_x + half_w
        top, bottom = -half_h, half_h

        in_hole = any(h.contains(x, y) for h in self.holes)
        if in_hole:
            # 孔内：距任一孔轮廓不超过边缘带 -> 不确定带，否则孔内可见
            for h in self.holes:
                if h.contains(x, y):
                    d = min(x - h.x0, h.x1 - x, y - h.y0, h.y1 - y)
                    if d <= self.edge_band_mm:
                        return EDGE_UNCERTAIN
                    return HOLE_VISIBLE

        inside_plate = left <= x <= right and top <= y <= bottom
        if inside_plate:
            d_out = min(x - left, right - x, y - top, bottom - y)
            if d_out <= self.edge_band_mm:
                return EDGE_UNCERTAIN
            return OCCLUDED

        # 板外：到板外轮廓的最近距离在边缘带内也算不确定
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        if (dx * dx + dy * dy) ** 0.5 <= self.edge_band_mm:
            return EDGE_UNCERTAIN
        return FREE


@dataclass
class Grid:
    """正交像素网格：像素 (row, col) 的射线固定在 (x_mm, y_mm)。"""
    width_px: int
    height_px: int
    view_w_mm: float
    view_h_mm: float

    @property
    def pixel_size_mm(self) -> float:
        return self.view_w_mm / self.width_px

    def pixel_to_ray(self, row: int, col: int) -> tuple[float, float]:
        """像素中心对应的世界系 (x, y)；像素为正方形。"""
        ps = self.pixel_size_mm
        x = -self.view_w_mm / 2.0 + (col + 0.5) * ps
        y = -self.view_h_mm / 2.0 + (row + 0.5) * ps
        return x, y

    def region_map(self, plate: Plate, center_x: float) -> list[list[str]]:
        """独立射线逐像素给出区域真值。"""
        out: list[list[str]] = []
        for r in range(self.height_px):
            row: list[str] = []
            for c in range(self.width_px):
                x, y = self.pixel_to_ray(r, c)
                row.append(plate.region_at(x, y, center_x))
            out.append(row)
        return out


def region_counts(region_map: list[list[str]]) -> dict[str, int]:
    counts = {name: 0 for name in REGIONS}
    for row in region_map:
        for label in row:
            counts[label] += 1
    return counts


def mask_of(region_map: list[list[str]], label: str) -> list[list[bool]]:
    return [[cell == label for cell in row] for row in region_map]


def union_masks(*masks: list[list[bool]]) -> list[list[bool]]:
    h, w = len(masks[0]), len(masks[0][0])
    return [[any(m[r][c] for m in masks) for c in range(w)] for r in range(h)]
