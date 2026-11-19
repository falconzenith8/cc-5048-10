"""实景遮挡合成内核。

相容深度定义：虚拟渲染深度与配准后的实景深度都在颜色相机坐标系下，
均为沿光轴的 Z（毫米），因此逐像素比较 z_virtual 与 z_real：

  z_virtual < z_real - bias  -> 虚拟在前（覆盖实景）
  z_virtual > z_real + bias  -> 现实在前（虚拟被前景遮住）
  |差值| <= bias             -> 共面/接近，不做武断判定

缺测（深度无效）不是「无限远」：
  - 虚拟覆盖到实景缺测像素时默认仍绘制虚拟，但该像素单独计入
    missing_count，前端用斜纹高亮，不允许把缺测错误隐藏成正常合成；
  - show_missing_warning=False 时仅关闭高亮，统计数字仍然保留。

近裁剪/远裁剪：被近裁剪面剔除的虚拟像素计入 near_clipped；超出远裁剪
面计入 far_virtual。透明现实物体（玻璃等）深度传感器通常无回波，
在本实现中等价为缺测，其限制在 README 中说明。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


RELATION_VIRTUAL_FRONT = "virtual_front"
RELATION_REAL_FRONT = "real_front"
RELATION_NEAR_PLANE = "near_clipped"
RELATION_FAR_PLANE = "far_virtual"
RELATION_MISSING = "real_missing"
RELATION_PLANAR = "near_planar"
RELATION_BACKGROUND = "background"


@dataclass
class CompositeResult:
    image: np.ndarray                  # (H,W,3) uint8
    relation_map: np.ndarray           # (H,W) uint8 关系标签码
    missing_mask: np.ndarray           # (H,W) bool 缺测且被虚拟覆盖
    margin_mm: np.ndarray              # (H,W) int16 实景-虚拟深度差，仅重叠区
    stats: dict = field(default_factory=dict)


def composite(
    color_rgb: np.ndarray,
    real_depth_mm: np.ndarray,
    real_valid: np.ndarray,
    virtual_depth_mm: np.ndarray,
    virtual_color: np.ndarray,
    virtual_coverage: np.ndarray,
    near_mm: float,
    far_mm: float,
    depth_bias_mm: float = 20.0,
    show_missing_warning: bool = True,
) -> CompositeResult:
    h, w = real_depth_mm.shape
    out = color_rgb.astype(np.float64).copy()
    relation = np.full((h, w), RELATION_BACKGROUND, dtype=object)
    missing = np.zeros((h, w), dtype=bool)
    margin = np.zeros((h, w), dtype=np.int16)

    v_cov = virtual_coverage
    v_z = virtual_depth_mm
    r_valid = real_valid & (real_depth_mm > 0)

    # 近裁剪诊断：覆盖区中出现极小深度（渲染器对 <= near 的面已剔除，
    # 这里保留接口，供渲染器输出的近裁剪标记扩展使用）。
    near_mask = v_cov & (v_z > 0) & (v_z < near_mm)
    far_mask = v_cov & (v_z > far_mm)

    overlap = v_cov & (v_z > 0) & r_valid
    only_virtual = v_cov & (v_z > 0) & ~r_valid

    diff = real_depth_mm - v_z          # 正：虚拟更近；负：实景更近
    v_front = overlap & (diff > depth_bias_mm)
    r_front = overlap & (diff < -depth_bias_mm)
    planar = overlap & ~v_front & ~r_front

    relation[overlap] = RELATION_PLANAR
    relation[v_front] = RELATION_VIRTUAL_FRONT
    relation[r_front] = RELATION_REAL_FRONT
    relation[only_virtual] = RELATION_MISSING
    relation[near_mask] = RELATION_NEAR_PLANE
    relation[far_mask] = RELATION_FAR_PLANE
    missing[only_virtual] = True

    out[v_front] = virtual_color[v_front] * 255.0
    # 共面像素做 50/50 混合，避免 z-fighting 式闪烁
    out[planar] = 0.5 * out[planar] + 0.5 * virtual_color[planar] * 255.0
    # 现实在前：保留实景颜色（即 out 不动）
    # 缺测区：绘制虚拟，但高亮由后处理叠加
    out[only_virtual] = virtual_color[only_virtual] * 255.0

    if show_missing_warning:
        yy, xx = np.where(missing)
        stripe = ((xx + yy) // 12) % 2 == 0
        sel = stripe
        rows, cols = yy[sel], xx[sel]
        out[rows, cols] = 0.6 * out[rows, cols] + 0.4 * np.array([255, 120, 0])

    margin_vals = np.clip(diff[overlap], -32768, 32767).astype(np.int16)
    margin[overlap] = margin_vals

    def count(mask: np.ndarray) -> int:
        return int(mask.sum())

    stats = {
        "virtual_coverage_px": count(v_cov),
        "overlap_px": count(overlap),
        "virtual_front_px": count(v_front),
        "real_front_px": count(r_front),
        "planar_px": count(planar),
        "missing_px": count(only_virtual),
        "near_clipped_px": count(near_mask),
        "far_virtual_px": count(far_mask),
        "real_invalid_total_px": int((~r_valid).sum()),
        "mean_margin_mm": float(diff[overlap].mean()) if overlap.any() else None,
        "min_margin_mm": float(diff[overlap].min()) if overlap.any() else None,
        "max_margin_mm": float(diff[overlap].max()) if overlap.any() else None,
        "near_mm": float(near_mm),
        "far_mm": float(far_mm),
        "depth_bias_mm": float(depth_bias_mm),
    }
    return CompositeResult(
        image=np.clip(out, 0, 255).astype(np.uint8),
        relation_map=relation,
        missing_mask=missing,
        margin_mm=margin,
        stats=stats,
    )
