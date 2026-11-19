"""带真值的模拟录制。

没有硬件时用本模块生成「录制格式」数据，而不是手画前景轮廓冒充深度：
- 颜色帧：真实场景网格渲染到颜色相机；
- 深度帧：同一真实场景渲染到另一台（有外参偏移的）深度相机，单位毫米，
  0 为无效值，并人为挖出一块缺测区域模拟远处缺测；
- 真值合成：把真实场景 + 虚拟展品放进同一世界系做单次 z-buffer 渲染，
  得到理想遮挡结果，供 selftest 核对边缘与空间位置；
- 元数据 JSON 记录两台相机内参、深度外参、深度单位/无效值和移动轨迹。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import geometry
from .calibration import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthCamera,
    decode_depth,
    look_at,
)
from .rasterize import Mesh, render_mesh

COLOR_W, COLOR_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480
NEAR_MM, FAR_MM = 150.0, 6000.0
DEPTH_SCALE = 1.0          # 原始深度单位 = 毫米
INVALID_VALUE = 0
REFERENCE_EXHIBIT = np.array([0.0, 0.0, 420.0])  # 展品参考世界坐标


@dataclass
class FrameBundle:
    index: int
    color: np.ndarray                 # (H,W,3) uint8
    depth_raw: np.ndarray             # (Hd,Wd) uint16，0 无效
    color_extrinsics: CameraExtrinsics
    depth_extrinsics: CameraExtrinsics
    color_K: CameraIntrinsics
    depth_K: CameraIntrinsics
    gt_composite: np.ndarray          # 参考位置下的理想合成 uint8
    real_only: np.ndarray             # 无虚拟物的纯实景颜色 uint8
    hole_rect: tuple[int, int, int, int]


def default_color_intrinsics() -> CameraIntrinsics:
    fx = fy = 525.0
    return CameraIntrinsics(COLOR_W, COLOR_H, fx, fy,
                            COLOR_W / 2 - 0.5, COLOR_H / 2 - 0.5)


def default_depth_intrinsics() -> CameraIntrinsics:
    fx = fy = 540.0
    return CameraIntrinsics(DEPTH_W, DEPTH_H, fx, fy,
                            DEPTH_W / 2 - 0.5, DEPTH_H / 2 - 0.5)


def camera_trajectory(n_frames: int) -> list[CameraExtrinsics]:
    """颜色相机沿圆弧移动，始终看向桌面中心。"""
    target = np.array([0.0, 50.0, 420.0])
    up = np.array([0.0, -1.0, 0.0])
    traj = []
    for i in range(n_frames):
        ang = -0.5 + 1.0 * i / max(n_frames - 1, 1)
        eye = np.array([900.0 * np.sin(ang), -1050.0,
                        420.0 + 900.0 * (1.0 - np.cos(ang))])
        traj.append(look_at(eye, target, up))
    return traj


def depth_pose_from_color(color_ext: CameraExtrinsics) -> CameraExtrinsics:
    """深度相机相对颜色相机固定偏移：右 30mm、下 15mm、后 40mm，朝向相同。

    偏移定义在颜色相机坐标系中，需经 R_c 旋转到世界系再叠加光心。
    """
    offset_c = np.array([30.0, 15.0, -40.0])
    depth_eye = color_ext.t + color_ext.R @ offset_c
    return CameraExtrinsics(R=color_ext.R.copy(), t=depth_eye)


def depth_camera_from_poses(color_ext: CameraExtrinsics,
                            depth_ext: CameraExtrinsics,
                            depth_K: CameraIntrinsics) -> DepthCamera:
    """由两机位姿求 T_cd（深度系 -> 颜色系）。

    p_c = R_c^T (R_d p_d + t_d - t_c)
        = (R_c^T R_d) p_d + R_c^T (t_d - t_c)
    """
    R_cd = color_ext.R.T @ depth_ext.R
    t_cd = color_ext.R.T @ (depth_ext.t - color_ext.t)
    return DepthCamera(intrinsics=depth_K, R_cd=R_cd, t_cd=t_cd,
                       depth_scale=DEPTH_SCALE, invalid_value=INVALID_VALUE)


def _shade(color_rgb: np.ndarray, coverage: np.ndarray, frame_idx: int) -> np.ndarray:
    img = (color_rgb * 255.0).astype(np.float64)
    rng = np.random.default_rng(1000 + frame_idx)
    noise = rng.normal(0, 0.8, img.shape)
    img = np.clip(img + noise, 0, 255)
    img[~coverage] = np.array([24, 24, 32])
    return img.astype(np.uint8)


def _carve_hole(depth_mm: np.ndarray, K: CameraIntrinsics,
                ext: CameraExtrinsics) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """在展品投影区左侧挖出一块矩形缺测，模拟深度传感器无回波。

    缺测位置选在真实场景中的实体上（桌面/墙区域），因此合成时虚拟像素
    会落在缺测区，触发显式缺测告警，而不是被悄悄当作无限远。
    """
    target_w = np.array([-180.0, 120.0, 420.0])
    p_c = ext.camera_from_world(target_w[None, :])[0]
    u = int(round(p_c[0] / p_c[2] * K.fx + K.cx))
    v = int(round(p_c[1] / p_c[2] * K.fy + K.cy))
    x0, y0 = max(u - 45, 0), max(v - 60, 0)
    x1, y1 = min(u + 45, K.width), min(v + 60, K.height)
    out = depth_mm.copy()
    out[y0:y1, x0:x1] = 0.0
    return out, (x0, y0, x1, y1)


def build_real_scene() -> Mesh:
    return geometry.merge_meshes([
        geometry.build_back_wall(),
        geometry.build_real_table(),
        geometry.build_foreground_board(),
    ])


def simulate_frame(index: int,
                   color_ext: CameraExtrinsics,
                   depth_ext: CameraExtrinsics,
                   color_K: CameraIntrinsics,
                   depth_K: CameraIntrinsics,
                   real_scene: Mesh,
                   exhibit: Mesh) -> FrameBundle:
    # 1) 深度相机渲染真实场景 -> 原始深度（毫米，uint16，0 无效）
    d_mm, _, d_cov = render_mesh(real_scene, depth_ext, depth_K, NEAR_MM, FAR_MM)
    d_mm_holed, hole = _carve_hole(d_mm, depth_K, depth_ext)
    depth_raw = np.where(d_cov, np.rint(d_mm_holed).astype(np.uint16), 0)

    # 2) 颜色相机渲染纯实景
    real_z, real_col, real_cov = render_mesh(real_scene, color_ext, color_K,
                                             NEAR_MM, FAR_MM)
    real_only = _shade(real_col, real_cov, index)

    # 3) 真值：真实场景 + 参考位置展品单次 z-buffer
    exhibit_ref = exhibit.translated(REFERENCE_EXHIBIT)
    full_scene = geometry.merge_meshes([real_scene, exhibit_ref])
    _, gt_col, gt_cov = render_mesh(full_scene, color_ext, color_K,
                                    NEAR_MM, FAR_MM)
    gt = _shade(gt_col, gt_cov, index)

    return FrameBundle(
        index=index, color=real_only, depth_raw=depth_raw,
        color_extrinsics=color_ext, depth_extrinsics=depth_ext,
        color_K=color_K, depth_K=depth_K,
        gt_composite=gt, real_only=real_only, hole_rect=hole,
    )


def simulate_recording(n_frames: int = 12) -> list[FrameBundle]:
    color_K = default_color_intrinsics()
    depth_K = default_depth_intrinsics()
    real_scene = build_real_scene()
    exhibit = geometry.build_exhibit()
    frames = []
    for i, color_ext in enumerate(camera_trajectory(n_frames)):
        depth_ext = depth_pose_from_color(color_ext)
        frames.append(simulate_frame(i, color_ext, depth_ext, color_K, depth_K,
                                     real_scene, exhibit))
    return frames


def _ext_to_json(ext: CameraExtrinsics) -> dict:
    return {"R": ext.R.tolist(), "t": ext.t.tolist()}


def save_recording(frames: list[FrameBundle], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cK = frames[0].color_K
    dK = frames[0].depth_K
    depth_cam = depth_camera_from_poses(
        frames[0].color_extrinsics, frames[0].depth_extrinsics, dK)
    meta = {
        "format": "occlude-recording-v1",
        "units": "millimeter",
        "depth_scale": DEPTH_SCALE,
        "depth_invalid_value": INVALID_VALUE,
        "color_intrinsics": {
            "width": cK.width, "height": cK.height, "fx": cK.fx, "fy": cK.fy,
            "cx": cK.cx, "cy": cK.cy,
        },
        "depth_intrinsics": {
            "width": dK.width, "height": dK.height, "fx": dK.fx, "fy": dK.fy,
            "cx": dK.cx, "cy": dK.cy,
        },
        "depth_to_color_extrinsics": {
            "R_cd": depth_cam.R_cd.tolist(),
            "t_cd": depth_cam.t_cd.tolist(),
        },
        "near_mm": NEAR_MM,
        "far_mm": FAR_MM,
        "reference_exhibit_xyz_mm": REFERENCE_EXHIBIT.tolist(),
        "frames": [],
    }
    for f in frames:
        name = f"{f.index:04d}"
        Image.fromarray(f.color).save(os.path.join(out_dir, f"color_{name}.png"))
        Image.fromarray(f.depth_raw).save(os.path.join(out_dir, f"depth_{name}.png"))
        Image.fromarray(f.gt_composite).save(
            os.path.join(out_dir, f"gt_{name}.png"))
        meta["frames"].append({
            "index": f.index,
            "color": f"color_{name}.png",
            "depth": f"depth_{name}.png",
            "ground_truth": f"gt_{name}.png",
            "hole_rect_xyxy": list(f.hole_rect),
            "color_extrinsics": _ext_to_json(f.color_extrinsics),
            "depth_extrinsics": _ext_to_json(f.depth_extrinsics),
        })
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def load_recording(in_dir: str) -> tuple[list[FrameBundle], dict]:
    with open(os.path.join(in_dir, "meta.json"), encoding="utf-8") as fp:
        meta = json.load(fp)
    ck = meta["color_intrinsics"]
    dk = meta["depth_intrinsics"]
    color_K = CameraIntrinsics(ck["width"], ck["height"], ck["fx"], ck["fy"],
                               ck["cx"], ck["cy"])
    depth_K = CameraIntrinsics(dk["width"], dk["height"], dk["fx"], dk["fy"],
                               dk["cx"], dk["cy"])
    frames = []
    for entry in meta["frames"]:
        color = np.asarray(Image.open(os.path.join(in_dir, entry["color"])).convert("RGB"))
        depth_raw = np.asarray(Image.open(os.path.join(in_dir, entry["depth"])))
        gt = np.asarray(Image.open(os.path.join(in_dir, entry["ground_truth"])).convert("RGB"))
        c_ext = CameraExtrinsics(np.array(entry["color_extrinsics"]["R"]),
                                 np.array(entry["color_extrinsics"]["t"]))
        d_ext = CameraExtrinsics(np.array(entry["depth_extrinsics"]["R"]),
                                 np.array(entry["depth_extrinsics"]["t"]))
        frames.append(FrameBundle(
            index=entry["index"], color=color, depth_raw=depth_raw.astype(np.uint16),
            color_extrinsics=c_ext, depth_extrinsics=d_ext,
            color_K=color_K, depth_K=depth_K, gt_composite=gt,
            real_only=color, hole_rect=tuple(entry["hole_rect_xyxy"]),
        ))
    return frames, meta
