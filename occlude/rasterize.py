"""CPU 光栅化器（z-buffer）。

虚拟物体与实景深度必须采用相容的深度定义：两者都在颜色相机坐标系下，
深度 = 沿光轴的 Z 距离（毫米），因此可逐像素直接比较。

近裁剪：三角面任一顶点 z <= near 即整个剔除（无裁剪平面细分），被裁剪
像素在合成阶段记为「近裁剪」诊断，而不是悄悄丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CameraExtrinsics, CameraIntrinsics


@dataclass
class Mesh:
    vertices: np.ndarray   # (N,3) 世界系，毫米
    faces: np.ndarray      # (M,3) 顶点索引
    colors: np.ndarray     # (N,3) 0..1 顶点色

    def translated(self, offset: np.ndarray) -> "Mesh":
        return Mesh(self.vertices + np.asarray(offset, dtype=np.float64),
                    self.faces, self.colors)


def render_mesh(
    mesh: Mesh,
    extrinsics: CameraExtrinsics,
    K: CameraIntrinsics,
    near_mm: float,
    far_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """渲染网格到颜色相机画面。

    返回：
      depth_mm  (H,W) float64，无覆盖处为 0
      color     (H,W,3) float64 0..1
      coverage  (H,W) bool
    """
    h, w = K.height, K.width
    pts_c = extrinsics.camera_from_world(mesh.vertices)
    z = pts_c[:, 2]
    proj_x = pts_c[:, 0] / np.where(z > 1e-6, z, np.nan) * K.fx + K.cx
    proj_y = pts_c[:, 1] / np.where(z > 1e-6, z, np.nan) * K.fy + K.cy

    depth_buf = np.full((h, w), np.inf, dtype=np.float64)
    color_buf = np.zeros((h, w, 3), dtype=np.float64)
    coverage = np.zeros((h, w), dtype=bool)

    for face in mesh.faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        z0, z1, z2 = z[i0], z[i1], z[i2]
        if min(z0, z1, z2) <= near_mm or max(z0, z1, z2) > far_mm * 1.05:
            continue
        p = np.array([[proj_x[i0], proj_y[i0]],
                      [proj_x[i1], proj_y[i1]],
                      [proj_x[i2], proj_y[i2]]])
        xmin = max(int(np.floor(p[:, 0].min())), 0)
        xmax = min(int(np.ceil(p[:, 0].max())), w - 1)
        ymin = max(int(np.floor(p[:, 1].min())), 0)
        ymax = min(int(np.ceil(p[:, 1].max())), h - 1)
        if xmin > xmax or ymin > ymax:
            continue

        px, py = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        # 像素中心采样
        sx = px + 0.5
        sy = py + 0.5

        def edge(ax, ay, bx, by):
            return (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)

        area = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(area) < 1e-9:
            continue
        w0 = edge(p[1, 0], p[1, 1], p[2, 0], p[2, 1]) / area
        w1 = edge(p[2, 0], p[2, 1], p[0, 0], p[0, 1]) / area
        w2 = edge(p[0, 0], p[0, 1], p[1, 0], p[1, 1]) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        # 透视校正插值：先除以各自 z，再用 1/z 加权还原
        inv_z = w0 / z0 + w1 / z1 + w2 / z2
        interp_z = 1.0 / inv_z
        interp_col = np.empty((*interp_z.shape, 3))
        for ch in range(3):
            c_over_z = (w0 * mesh.colors[i0, ch] / z0
                        + w1 * mesh.colors[i1, ch] / z1
                        + w2 * mesh.colors[i2, ch] / z2)
            interp_col[..., ch] = c_over_z * interp_z

        rows = py[inside]
        cols = px[inside]
        pz = interp_z[inside]
        pc = interp_col[inside]
        nearer = pz < depth_buf[rows, cols]
        r2, c2 = rows[nearer], cols[nearer]
        depth_buf[r2, c2] = pz[nearer]
        color_buf[r2, c2] = pc[nearer]
        coverage[r2, c2] = True

    depth_out = np.where(coverage, depth_buf, 0.0)
    return depth_out, np.clip(color_buf, 0.0, 1.0), coverage
