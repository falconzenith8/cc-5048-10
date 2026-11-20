"""标准测试想定：虚拟展品、带孔薄板、显式无效区与扫动轨迹。

想定中特意布置两类「无数据」像素，规则与主项目 README 一致——
深度无效由有效位标志决定，深度数值 0 本身不携带语义：

1. invalid_patch（零深度无效区）：传感器在该矩形内明确给出 valid=False、
   depth=0。它与一片真实位于近处的表面 near_adjacent 直接相邻（只隔
   2mm 间隙），用于检验合成器是否遵守「缺测按声明规则判定，不得一律
   当成最近遮挡物」。
2. 板实体与边缘带本身不产生无效位；它们只是普通遮挡，深度为板深度，
   保证「缺测」与「被近处实体遮挡」两条故障路径不会混入同一区域。
"""
from __future__ import annotations

from dataclasses import dataclass

from depth_rig.geometry import Grid, Hole, Plate, Rect

# 正交视场（毫米）与像素分辨率
VIEW_W_MM = 200.0
VIEW_H_MM = 160.0
WIDTH_PX = 80
HEIGHT_PX = 64

# 帧与时间戳
FRAME_PERIOD_S = 1.0 / 30.0
N_FRAMES = 36

# 薄板（位于展品与相机之间）
PLATE_Z_MM = 400.0
PLATE_W_MM = 90.0
PLATE_H_MM = 120.0
EDGE_BAND_MM = 1.0

# 板横向扫动：每帧位移（毫米）。fast 为基准速度，slow 用于回放对照。
# slow 每帧位移仍大于一个像素（2.5mm），保证边缘扫动像素非空但显著更少。
PLATE_SPEED_FAST_MM_PER_FRAME = 18.0
PLATE_SPEED_SLOW_MM_PER_FRAME = 3.75
PLATE_X_AT_T0 = -190.0          # 第一帧板中心在视场左侧外


@dataclass(frozen=True)
class Scenario:
    grid: Grid
    plate: Plate
    surfaces: tuple[Rect, ...]
    invalid_patches: tuple[Rect, ...]
    n_frames: int
    frame_period_s: float
    speed_mm_per_frame: float
    plate_x_t0: float

    def plate_center_x(self, frame_index: int,
                       speed_mm_per_frame: float | None = None) -> float:
        v = self.speed_mm_per_frame if speed_mm_per_frame is None else speed_mm_per_frame
        return self.plate_x_t0 + v * frame_index

    def timestamp_s(self, frame_index: int) -> float:
        return frame_index * self.frame_period_s


def build_scenario(speed_mm_per_frame: float = PLATE_SPEED_FAST_MM_PER_FRAME,
                   plate_z_mm: float = PLATE_Z_MM) -> Scenario:
    grid = Grid(WIDTH_PX, HEIGHT_PX, VIEW_W_MM, VIEW_H_MM)

    # 虚拟展品：三层与相机平面平行的表面，颜色按层区分，便于逐像素核对
    back_wall = Rect(
        "back_wall", -100.0, -80.0, 100.0, 80.0, z=1600.0,
        color=(70, 90, 130))
    mid_surface = Rect(
        "mid_surface", -60.0, 12.0, -12.0, 42.0, z=900.0,
        color=(90, 170, 110))
    near_adjacent = Rect(
        "near_adjacent", 60.0, 30.0, 88.0, 52.0, z=520.0,
        color=(220, 180, 90))
    surfaces = (back_wall, mid_surface, near_adjacent)

    # 零深度但被明确标为无效的区域：右下方，与近表面仅隔 2mm 间隙
    invalid_patch = Rect(
        "invalid_zero_patch", 60.0, 54.0, 88.0, 72.0, z=0.0,
        color=(255, 0, 255))
    # 远处一枚孤立无效像素，验证规则不只对连续大块生效
    invalid_isolated = Rect(
        "invalid_isolated", 80.0, -60.0, 82.5, -57.5, z=0.0,
        color=(255, 0, 255))
    invalid_patches = (invalid_patch, invalid_isolated)

    # 板上两孔，刻意分离两类深度：
    # hole_far 只望向远墙（深 >=1600，单位错误区）；
    # hole_mid 只望向中层表面（900，单位错误阈值之下），二者不重叠，
    # 因此「单位错误」与「孔内补深度」的失败像素从几何上就互不相交。
    holes = (
        Hole("hole_far", -14.0, -44.0, 10.0, -22.0),
        Hole("hole_mid", -54.0, 16.0, -18.0, 38.0),
    )
    plate = Plate(
        width_mm=PLATE_W_MM, height_mm=PLATE_H_MM,
        z_plate_mm=plate_z_mm, holes=holes,
        edge_band_mm=EDGE_BAND_MM)

    return Scenario(
        grid=grid, plate=plate, surfaces=surfaces,
        invalid_patches=invalid_patches,
        n_frames=N_FRAMES, frame_period_s=FRAME_PERIOD_S,
        speed_mm_per_frame=speed_mm_per_frame,
        plate_x_t0=PLATE_X_AT_T0)
