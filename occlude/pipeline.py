"""端到端管线门面：录制帧 + 虚拟物位姿 -> 诊断结果。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import geometry
from .calibration import (
    CameraExtrinsics,
    decode_depth,
    register_depth_to_color,
)
from .composite import composite
from .dataset import (
    FAR_MM,
    INVALID_VALUE,
    NEAR_MM,
    REFERENCE_EXHIBIT,
    FrameBundle,
    depth_camera_from_poses,
)
from .edge import diagnose_registration
from .rasterize import render_mesh

# 虚拟展品允许的世界系移动范围（毫米），由实景遮挡内核统一校验
MOVE_LIMITS = {
    "x": (-700.0, 700.0),
    "y": (-200.0, 400.0),
    "z": (-300.0, 900.0),
}


class PipelineError(ValueError):
    """内核拒绝的请求（越界位姿等），前端需原样反馈。"""


@dataclass
class Evaluation:
    composite_rgb: np.ndarray
    depth_view_rgb: np.ndarray
    mask_view_rgb: np.ndarray
    edge_overlay_rgb: np.ndarray
    relation_codes: np.ndarray       # uint8
    registered_depth_mm: np.ndarray
    registered_valid: np.ndarray
    virtual_depth_mm: np.ndarray
    stats: dict
    edge: dict
    warnings: list[str]
    exhibit_xyz: np.ndarray
    frame_index: int


RELATION_CODE = {
    "background": 0,
    "virtual_front": 1,
    "real_front": 2,
    "near_planar": 3,
    "real_missing": 4,
    "near_clipped": 5,
    "far_virtual": 6,
}
RELATION_NAME = {v: k for k, v in RELATION_CODE.items()}


def validate_offset(offset: np.ndarray) -> None:
    for axis, val in zip(("x", "y", "z"), offset):
        lo, hi = MOVE_LIMITS[axis]
        if not (lo <= float(val) <= hi):
            raise PipelineError(
                f"虚拟物 {axis} 轴偏移 {float(val):.0f}mm 超出内核允许范围 "
                f"[{lo:.0f}, {hi:.0f}]mm，请求被拒绝（未执行合成）。")


def _depth_colormap(depth_mm: np.ndarray, valid: np.ndarray,
                    far_mm: float) -> np.ndarray:
    h, w = depth_mm.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if valid.any():
        d = depth_mm[valid]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth_mm - lo) / (hi - lo), 0, 1)
        # 近=暖色，远=冷色
        img[..., 0] = (255 * (1 - norm) * valid).astype(np.uint8)
        img[..., 1] = (180 * (1 - np.abs(0.5 - norm) * 2) * valid).astype(np.uint8)
        img[..., 2] = (255 * norm * valid).astype(np.uint8)
    img[~valid] = np.array([40, 40, 40])
    return img


def _mask_colormap(valid: np.ndarray) -> np.ndarray:
    img = np.zeros((*valid.shape, 3), dtype=np.uint8)
    img[valid] = (90, 200, 120)
    img[~valid] = (200, 70, 60)
    return img


def evaluate_frame(frame: FrameBundle,
                   exhibit_offset: np.ndarray,
                   depth_bias_mm: float = 10.0,
                   show_missing_warning: bool = True,
                   edge_tolerance_px: float = 3.0) -> Evaluation:
    offset = np.asarray(exhibit_offset, dtype=np.float64)
    validate_offset(offset)
    warnings: list[str] = []

    depth_cam = depth_camera_from_poses(
        frame.color_extrinsics, frame.depth_extrinsics, frame.depth_K)
    depth_mm = decode_depth(frame.depth_raw, depth_cam.depth_scale,
                            depth_cam.invalid_value)

    # 完整三维配准：深度像素 -> 深度系点云 -> 颜色系 -> 颜色像素
    reg_depth, reg_valid = register_depth_to_color(
        depth_mm, frame.depth_K, depth_cam, frame.color_K)

    exhibit = geometry.build_exhibit().translated(REFERENCE_EXHIBIT + offset)
    v_depth, v_color, v_cov = render_mesh(
        exhibit, frame.color_extrinsics, frame.color_K, NEAR_MM, FAR_MM)

    result = composite(
        color_rgb=frame.real_only,
        real_depth_mm=reg_depth,
        real_valid=reg_valid,
        virtual_depth_mm=v_depth,
        virtual_color=v_color,
        virtual_coverage=v_cov,
        near_mm=NEAR_MM,
        far_mm=FAR_MM,
        depth_bias_mm=depth_bias_mm,
        show_missing_warning=show_missing_warning,
    )
    codes = np.zeros(result.relation_map.shape, dtype=np.uint8)
    for name, code in RELATION_CODE.items():
        codes[result.relation_map == name] = code

    edge_diag = diagnose_registration(
        frame.real_only, reg_depth, reg_valid, tolerance_px=edge_tolerance_px)

    if result.stats["missing_px"] > 0:
        warnings.append(
            f"{result.stats['missing_px']} 个虚拟像素落在实景深度缺测区，"
            "已按缺测策略绘制并高亮，不能视为可靠遮挡。")
    if result.stats["near_clipped_px"] > 0:
        warnings.append(
            f"{result.stats['near_clipped_px']} 个虚拟像素触发近裁剪 "
            f"(near={NEAR_MM:.0f}mm)。")
    if result.stats["far_virtual_px"] > 0:
        warnings.append(
            f"{result.stats['far_virtual_px']} 个虚拟像素超出远裁剪面 "
            f"(far={FAR_MM:.0f}mm)。")
    edge_dict = edge_diag.as_dict()
    if edge_dict["match_ratio"] < 0.6:
        warnings.append(
            f"边缘配准匹配率仅 {edge_dict['match_ratio']:.1%}，"
            "请检查深度-颜色外参，遮挡边界可能存在系统性偏差。")

    return Evaluation(
        composite_rgb=result.image,
        depth_view_rgb=_depth_colormap(reg_depth, reg_valid, FAR_MM),
        mask_view_rgb=_mask_colormap(reg_valid),
        edge_overlay_rgb=edge_diag.overlay,
        relation_codes=codes,
        registered_depth_mm=reg_depth,
        registered_valid=reg_valid,
        virtual_depth_mm=v_depth,
        stats=result.stats,
        edge=edge_dict,
        warnings=warnings,
        exhibit_xyz=REFERENCE_EXHIBIT + offset,
        frame_index=frame.index,
    )


def encode_png(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
