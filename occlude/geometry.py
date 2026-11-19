"""程序化三角网格生成（单位：毫米）。"""
from __future__ import annotations

import numpy as np

from .rasterize import Mesh


def _finalize(verts: list[np.ndarray], faces: list[tuple[int, int, int]],
              color: tuple[float, float, float]) -> Mesh:
    v = np.stack(verts).astype(np.float64)
    c = np.tile(np.array(color, dtype=np.float64), (len(verts), 1))
    return Mesh(vertices=v, faces=np.array(faces, dtype=np.int64), colors=c)


def box(center: np.ndarray, size: np.ndarray,
        color: tuple[float, float, float]) -> Mesh:
    """轴对齐长方体。size = (sx, sy, sz) 全边长。"""
    cx, cy, cz = center
    hx, hy, hz = np.asarray(size) / 2.0
    pts = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    f = [
        (0, 3, 2), (0, 2, 1),       # 后面 -z
        (4, 5, 6), (4, 6, 7),       # 前面 +z
        (0, 1, 5), (0, 5, 4),       # 顶面 -y
        (2, 3, 7), (2, 7, 6),       # 底面 +y
        (0, 4, 7), (0, 7, 3),       # 左面 -x
        (1, 2, 6), (1, 6, 5),       # 右面 +x
    ]
    return _finalize([np.array(p) for p in pts], f, color)


def cylinder(center: np.ndarray, radius: float, height: float,
             color: tuple[float, float, float], segments: int = 32) -> Mesh:
    """沿 Y 轴的圆柱（含上下盖）。"""
    cx, cy, cz = center
    y0, y1 = cy - height / 2, cy + height / 2
    verts: list[np.ndarray] = []
    for ring_y in (y0, y1):
        for k in range(segments):
            a = 2 * np.pi * k / segments
            verts.append(np.array([cx + radius * np.cos(a), ring_y,
                                   cz + radius * np.sin(a)]))
    bottom_center = len(verts)
    verts.append(np.array([cx, y0, cz]))
    top_center = len(verts)
    verts.append(np.array([cx, y1, cz]))

    faces: list[tuple[int, int, int]] = []
    for k in range(segments):
        kn = (k + 1) % segments
        b0, b1 = k, kn
        t0, t1 = segments + k, segments + kn
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
        faces.append((bottom_center, b1, b0))
        faces.append((top_center, t0, t1))
    return _finalize(verts, faces, color)


def sphere(center: np.ndarray, radius: float,
           color: tuple[float, float, float], rings: int = 12,
           sectors: int = 20) -> Mesh:
    cx, cy, cz = center
    verts: list[np.ndarray] = []
    for r in range(rings + 1):
        theta = np.pi * r / rings
        for s in range(sectors):
            phi = 2 * np.pi * s / sectors
            verts.append(np.array([
                cx + radius * np.sin(theta) * np.cos(phi),
                cy + radius * np.cos(theta),
                cz + radius * np.sin(theta) * np.sin(phi),
            ]))
    faces: list[tuple[int, int, int]] = []
    for r in range(rings):
        for s in range(sectors):
            sn = (s + 1) % sectors
            a = r * sectors + s
            b = r * sectors + sn
            c = (r + 1) * sectors + sn
            d = (r + 1) * sectors + s
            faces.append((a, b, c))
            faces.append((a, c, d))
    return _finalize(verts, faces, color)


def merge_meshes(meshes: list[Mesh]) -> Mesh:
    verts, cols, faces = [], [], []
    offset = 0
    for m in meshes:
        verts.append(m.vertices)
        cols.append(m.colors)
        faces.append(m.faces + offset)
        offset += m.vertices.shape[0]
    return Mesh(np.concatenate(verts), np.concatenate(faces), np.concatenate(cols))


def build_exhibit() -> Mesh:
    """虚拟展品：三层塔状结构（底座 + 柱体 + 顶球），层间颜色不同，
    便于在参考合成中核对深度排序与边缘。"""
    base = box(np.array([0, 0, 0]), np.array([220, 60, 220]), (0.15, 0.45, 0.85))
    pole = cylinder(np.array([0, 110, 0]), 55, 160, (0.95, 0.75, 0.2), segments=28)
    top = sphere(np.array([0, 250, 0]), 70, (0.85, 0.25, 0.3), rings=12, sectors=24)
    return merge_meshes([base, pole, top])


def build_real_table() -> Mesh:
    """实景桌面：世界系中固定的真实桌子（仅用于模拟数据渲染真值）。

    桌面顶面（y=0）与展品底座底面齐平，展品正好落在桌面上。
    """
    top = box(np.array([0, 40, 250]), np.array([1600, 80, 1400]),
              (0.55, 0.42, 0.30))
    leg_h = 460
    leg_pts = [(-700, -350, -350), (700, -350, -350),
               (-700, -350, 850), (700, -350, 850)]
    legs = [box(np.array([x, 80 + leg_h / 2, z]),
                np.array([70, leg_h, 70]), (0.40, 0.30, 0.22))
            for x, _, z in leg_pts]
    return merge_meshes([top, *legs])


def build_foreground_board() -> Mesh:
    """实景前景遮挡板：竖在相机与展品之间、偏向一侧的红色板，
    只遮挡展品的一部分，用于验证真实前景遮挡。"""
    return box(np.array([60, -190, 430]), np.array([200, 120, 160]),
               (0.80, 0.12, 0.10))


def build_back_wall() -> Mesh:
    """远处墙面带拼接色块，提供可对齐的颜色纹理边缘。"""
    wall = box(np.array([0, 200, 1500]), np.array([2600, 1600, 60]),
               (0.72, 0.74, 0.78))
    panels = [
        (np.array([-520, 260, 1465]), np.array([360, 220, 20]), (0.30, 0.45, 0.65)),
        (np.array([520, 260, 1465]), np.array([360, 220, 20]), (0.65, 0.55, 0.30)),
        (np.array([0, -120, 1465]), np.array([420, 200, 20]), (0.40, 0.60, 0.45)),
        (np.array([-760, -260, 1465]), np.array([260, 260, 20]), (0.55, 0.40, 0.60)),
        (np.array([760, -260, 1465]), np.array([260, 260, 20]), (0.62, 0.42, 0.38)),
    ]
    extras = [box(c, s, col) for c, s, col in panels]
    return merge_meshes([wall, *extras])
