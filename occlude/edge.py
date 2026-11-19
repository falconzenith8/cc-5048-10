"""边缘配准诊断。

深度配准到颜色相机后，深度边缘应与颜色纹理边缘对齐。本模块提取两侧
边缘并计算倒角距离（毫米像素单位），把系统性配准偏差变成可见诊断，
而不是让它静默污染遮挡判定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T

    def conv3(img: np.ndarray, k: np.ndarray) -> np.ndarray:
        p = np.pad(img, 1, mode="edge")
        acc = np.zeros_like(img)
        for j in range(3):
            for i in range(3):
                acc += k[j, i] * p[j:j + img.shape[0], i:i + img.shape[1]]
        return acc

    return np.hypot(conv3(g, kx), conv3(g, ky))


def _remove_isolated(mask: np.ndarray) -> np.ndarray:
    """3x3 多数滤波：压制孤立噪声点，保留连续几何/纹理边缘。"""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dr in range(3):
        for dc in range(3):
            count += p[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return count >= 3


def _box_blur_3(g: np.ndarray) -> np.ndarray:
    p = np.pad(g, 1, mode="edge")
    acc = np.zeros_like(g)
    for dr in range(3):
        for dc in range(3):
            acc += p[dr:dr + g.shape[0], dc:dc + g.shape[1]]
    return acc / 9.0


def color_edges(rgb: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    gray = _box_blur_3(rgb.mean(axis=2))
    mag = _sobel_magnitude(gray)
    thr = np.percentile(mag, percentile)
    raw = mag >= max(thr, 1e-6)
    return _remove_isolated(raw)


def depth_edges(depth_mm: np.ndarray, valid: np.ndarray,
                jump_mm: float = 60.0) -> np.ndarray:
    """深度边缘 = 两个有效像素之间的深度跳变。

    有效/无效边界属于缺测诊断，不属于配准边缘，这里不计入，避免缺测孔洞
    污染配准偏差统计。
    """
    d = depth_mm.copy()
    d[~valid] = 0.0
    right = np.zeros_like(d)
    down = np.zeros_like(d)
    both_h = valid[:, 1:] & valid[:, :-1]
    both_v = valid[1:, :] & valid[:-1, :]
    right[:, :-1] = np.where(both_h, np.abs(d[:, 1:] - d[:, :-1]), 0.0)
    down[:-1, :] = np.where(both_v, np.abs(d[1:, :] - d[:-1, :]), 0.0)
    return (right > jump_mm) | (down > jump_mm)


def chamfer_distance(edges: np.ndarray) -> np.ndarray:
    """两趟倒角距离变换（3x3 邻域，单位像素），无 scipy 依赖。"""
    h, w = edges.shape
    big = np.float64(h + w)
    dist = np.where(edges, 0.0, big)
    fwd = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, big], [big, big, big]])
    bwd = np.array([[big, big, big], [big, 0.0, 1.0], [1.0, 1.0, 1.0]])

    def pass_(d: np.ndarray, kernel: np.ndarray, reverse: bool) -> np.ndarray:
        rows = range(h - 1, -1, -1) if reverse else range(h)
        cols = range(w - 1, -1, -1) if reverse else range(w)
        for r in rows:
            for c in cols:
                best = d[r, c]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        kv = kernel[dr + 1, dc + 1]
                        if kv >= big:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            cand = d[rr, cc] + kv
                            if cand < best:
                                best = cand
                d[r, c] = best
        return d

    dist = pass_(dist, fwd, False)
    dist = pass_(dist, bwd, True)
    return dist


@dataclass
class EdgeDiagnostics:
    overlay: np.ndarray          # (H,W,3) uint8 颜色底 + 双色边缘
    color_edge_count: int
    depth_edge_count: int
    matched_depth_edges: int     # 距离 <= tolerance 的深度边缘像素
    mean_dist_px: float
    p95_dist_px: float
    max_dist_px: float
    tolerance_px: float

    def as_dict(self) -> dict:
        return {
            "color_edge_px": self.color_edge_count,
            "depth_edge_px": self.depth_edge_count,
            "matched_depth_edge_px": self.matched_depth_edges,
            "match_ratio": (self.matched_depth_edges / self.depth_edge_count
                            if self.depth_edge_count else 0.0),
            "mean_dist_px": self.mean_dist_px,
            "p95_dist_px": self.p95_dist_px,
            "max_dist_px": self.max_dist_px,
            "tolerance_px": self.tolerance_px,
        }


def diagnose_registration(
    color_rgb: np.ndarray,
    registered_depth_mm: np.ndarray,
    valid: np.ndarray,
    tolerance_px: float = 3.0,
) -> EdgeDiagnostics:
    c_edge = color_edges(color_rgb)
    d_edge = depth_edges(registered_depth_mm, valid)

    # 主指标：每条深度边缘到最近颜色边缘的距离（深度边缘几何含义明确）
    dist_to_color = chamfer_distance(c_edge)
    dists = dist_to_color[d_edge]
    if dists.size:
        mean = float(dists.mean())
        p95 = float(np.percentile(dists, 95))
        mx = float(dists.max())
        matched = int((dists <= tolerance_px).sum())
    else:
        mean = p95 = mx = 0.0
        matched = 0

    overlay = color_rgb.copy()
    overlay[d_edge & ~c_edge] = np.array([0, 180, 255])    # 深度边缘：青
    overlay[c_edge & ~d_edge] = np.array([255, 220, 0])    # 颜色边缘：黄
    overlay[c_edge & d_edge] = np.array([80, 255, 80])     # 重合：绿

    return EdgeDiagnostics(
        overlay=overlay.astype(np.uint8),
        color_edge_count=int(c_edge.sum()),
        depth_edge_count=int(d_edge.sum()),
        matched_depth_edges=matched,
        mean_dist_px=mean,
        p95_dist_px=p95,
        max_dist_px=mx,
        tolerance_px=float(tolerance_px),
    )
