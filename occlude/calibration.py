"""标定与坐标变换模块。

坐标约定：相机坐标系为 OpenCV 约定（X 右、Y 下、Z 朝前），深度即沿光轴
的 Z 距离（毫米）。世界系与首帧颜色相机系一致。

深度与颜色是两台独立相机：
- 各自有内参 (fx, fy, cx, cy)；
- 深度相对颜色相机的外参 T_cd（把深度系点变换到颜色系）单独处理；
- 配准必须经过「深度像素 -> 深度系三维 -> 颜色系 -> 颜色像素」的完整
  三维变换，不能直接把深度像素坐标当成颜色图坐标使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """相机在世界系中的位姿：R 列向量为相机轴在世界系中的方向。"""

    R: np.ndarray  # (3,3) 正交
    t: np.ndarray  # (3,) 光心世界坐标

    def pose_matrix(self) -> np.ndarray:
        """世界系 -> 相机系的 4x4 变换。"""
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, 3] = -self.R.T @ self.t
        return T

    def camera_from_world(self, points_w: np.ndarray) -> np.ndarray:
        return (self.R.T @ (points_w - self.t).T).T


@dataclass(frozen=True)
class DepthCamera:
    intrinsics: CameraIntrinsics
    """深度相对颜色相机的外参 T_cd：p_color = R_cd @ p_depth + t_cd。"""
    R_cd: np.ndarray
    t_cd: np.ndarray
    depth_scale: float = 1.0          # 原始单位 -> 毫米的倍率
    invalid_value: float = 0.0        # 原始单位下的无效标记值

    def transform_to_color(self, points_d: np.ndarray) -> np.ndarray:
        return (self.R_cd @ points_d.T).T + self.t_cd


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> CameraExtrinsics:
    """由视点/目标/向上向量构造 OpenCV 约定外参（Z 朝场景前方）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return CameraExtrinsics(R=R, t=eye.copy())


def invert_extrinsics(ext: CameraExtrinsics) -> tuple[np.ndarray, np.ndarray]:
    """返回外参的逆（相机系 -> 世界系）：R_cw, t_cw。"""
    return ext.R.copy(), ext.t.copy()


def project_points(
    points_c: np.ndarray, K: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """相机系点 -> 像素坐标与深度（沿光轴 Z，毫米）。"""
    z = points_c[:, 2]
    valid = z > 1e-6
    u = np.full(points_c.shape[0], np.nan)
    v = np.full(points_c.shape[0], np.nan)
    u[valid] = points_c[valid, 0] / z[valid] * K.fx + K.cx
    v[valid] = points_c[valid, 1] / z[valid] * K.fy + K.cy
    return np.stack([u, v], axis=1), z


def backproject_depth(
    depth_mm: np.ndarray, K: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """深度图（毫米）反投影为相机系点云，返回点云和有效像素 (v,u) 索引。"""
    h, w = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    valid = np.ones((h, w), dtype=bool)
    pts = np.stack(
        [
            (uu - K.cx) * depth_mm / K.fx,
            (vv - K.cy) * depth_mm / K.fy,
            depth_mm,
        ],
        axis=-1,
    )
    return pts, valid


def decode_depth(
    raw: np.ndarray, depth_scale: float, invalid_value: float
) -> np.ndarray:
    """原始深度 -> 毫米浮点深度；无效/非正值置 0。"""
    raw = raw.astype(np.float64)
    out = raw * float(depth_scale)
    if np.isfinite(invalid_value):
        out[np.isclose(raw, invalid_value)] = 0.0
    out[~np.isfinite(out)] = 0.0
    out[out <= 0.0] = 0.0
    return out


def register_depth_to_color(
    depth_mm: np.ndarray,
    depth_K: CameraIntrinsics,
    depth_cam: DepthCamera,
    color_K: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """把深度相机视差深度配准到颜色相机网格。

    返回 (registered_mm, valid_mask)。
    有效掩码完全来自深度有效像素；落在颜色画面外的点被丢弃，不做插值/
    孔洞填充——缺测区域会在合成阶段显式呈现，绝不隐藏。
    同一颜色像素被多个深度点命中时保留更近者（z-buffer 思想）。
    """
    h, w = depth_mm.shape
    pts_d, _ = backproject_depth(depth_mm, depth_K)
    pts_c = depth_cam.transform_to_color(pts_d.reshape(-1, 3))

    z_c = pts_c[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = pts_c[:, 0] / np.where(z_c > 1e-6, z_c, np.nan) * color_K.fx + color_K.cx
        v = pts_c[:, 1] / np.where(z_c > 1e-6, z_c, np.nan) * color_K.fy + color_K.cy
    safe = np.isfinite(u) & np.isfinite(v)
    ui = np.zeros_like(u, dtype=np.int64)
    vi = np.zeros_like(v, dtype=np.int64)
    ui[safe] = np.rint(u[safe]).astype(np.int64)
    vi[safe] = np.rint(v[safe]).astype(np.int64)

    valid_src = depth_mm.reshape(-1) > 0.0
    in_frame = (
        (ui >= 0) & (ui < color_K.width) & (vi >= 0) & (vi < color_K.height)
        & (z_c > 1e-6) & safe
    )
    keep = valid_src & in_frame

    registered = np.zeros((color_K.height, color_K.width), dtype=np.float64)
    valid = np.zeros((color_K.height, color_K.width), dtype=bool)
    order = np.argsort(-z_c)  # 远的先写，近的后写覆盖
    for idx in order:
        if not keep[idx]:
            continue
        col, row = ui[idx], vi[idx]
        if not valid[row, col] or z_c[idx] < registered[row, col]:
            registered[row, col] = z_c[idx]
            valid[row, col] = True
    return registered, valid


def encode_depth_mm(depth_mm: np.ndarray, invalid_value: float = 0.0) -> np.ndarray:
    """毫米深度 -> uint16 原始单位（1 深度单位 = 1 毫米）。"""
    out = np.rint(depth_mm).astype(np.uint16)
    out[depth_mm <= 0] = int(invalid_value)
    return out
