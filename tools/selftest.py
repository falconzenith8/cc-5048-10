"""内核自检：在带真值的模拟录制上验证遮挡、配准与错误路径。

运行：python3 tools/selftest.py
退出码 0 表示全部检查通过。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from occlude import MOVE_LIMITS, PipelineError, evaluate_frame, simulate_recording
from occlude.calibration import (
    CameraExtrinsics,
    decode_depth,
    register_depth_to_color,
)
from occlude.dataset import REFERENCE_EXHIBIT, depth_camera_from_poses
from occlude.edge import diagnose_registration
from occlude.pipeline import RELATION_CODE

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


def main() -> int:
    frames = simulate_recording(n_frames=8)
    check("录制帧数", len(frames) == 8, f"got {len(frames)}")

    # ---- 1. 外参往返：T_cd 把深度光心映射到颜色系中的深度光心 ----
    f0 = frames[0]
    dcam = depth_camera_from_poses(f0.color_extrinsics, f0.depth_extrinsics,
                                   f0.depth_K)
    origin_d = np.zeros((1, 3))
    mapped = dcam.transform_to_color(origin_d)[0]
    # T_cd 的平移项：深度光心在颜色系中的坐标 = R_c^T (t_d - t_c)
    expect = f0.color_extrinsics.R.T @ (f0.depth_extrinsics.t
                                        - f0.color_extrinsics.t)
    err = np.linalg.norm(mapped - expect)
    check("深度->颜色外参往返", err < 1e-6, f"err={err:.2e}mm")

    # ---- 2. 配准不能是恒等映射：深度像素坐标 != 颜色像素坐标 ----
    depth_mm = decode_depth(f0.depth_raw, dcam.depth_scale, dcam.invalid_value)
    reg, valid = register_depth_to_color(depth_mm, f0.depth_K, dcam, f0.color_K)
    shifted = np.abs(dcam.R_cd - np.eye(3)).sum() + np.abs(dcam.t_cd).sum()
    check("深度/颜色相机外参非平凡", shifted > 1e-6, f"offset norm={shifted:.2f}")
    check("配准输出尺寸=颜色尺寸", reg.shape == (f0.color_K.height,
                                                  f0.color_K.width))
    check("配准保留缺测（无孔洞填充）",
          valid.sum() < depth_mm.size and (reg[~valid] == 0).all(),
          f"valid={valid.sum()}/{valid.size}")

    # ---- 3. 参考位置下与真值合成核对 ----
    ev = evaluate_frame(f0, np.zeros(3))
    gt = f0.gt_composite.astype(np.int16)
    ours = ev.composite_rgb.astype(np.int16)
    # 只在 GT 与我方都有内容的区域比较；缺测斜纹区与 GT 本就不同，排除
    diff = np.abs(gt - ours).mean(axis=2)
    core = ev.relation_codes != RELATION_CODE["real_missing"]
    mae = float(diff[core].mean())
    mismatch_ratio = float((diff[core] > 40).mean())
    check("参考合成 vs 真值 平均色差", mae < 25.0, f"MAE={mae:.2f}")
    check("参考合成 vs 真值 严重不一致像素占比", mismatch_ratio < 0.05,
          f"{mismatch_ratio:.2%}")

    # ---- 4. 深度关系方向：前景遮挡板处应判现实在前 ----
    real_front = ev.relation_codes == RELATION_CODE["real_front"]
    virt_front = ev.relation_codes == RELATION_CODE["virtual_front"]
    check("存在现实在前像素（前景遮挡）", real_front.sum() > 500,
          f"{real_front.sum()}px")
    check("存在虚拟在前像素", virt_front.sum() > 500, f"{virt_front.sum()}px")
    # 展品底座在桌面之上：虚拟像素深度必须小于其配准实景深度
    check("深度差符号统计合理",
          ev.stats["min_margin_mm"] is not None
          and ev.stats["min_margin_mm"] < 0 < ev.stats["max_margin_mm"],
          f"margin [{ev.stats['min_margin_mm']}, {ev.stats['max_margin_mm']}]mm")

    # ---- 5. 移动展品：向左移出遮挡板后，现实在前像素应骤降、虚拟在前上升 ----
    ev_side = evaluate_frame(f0, np.array([-300.0, 0.0, 0.0]))
    moved_away = (
        ev_side.stats["real_front_px"] < ev.stats["real_front_px"] * 0.1
        and ev_side.stats["virtual_front_px"] > ev.stats["virtual_front_px"]
    )
    check("移动展品改变遮挡像素分布", moved_away,
          f"real_front {ev.stats['real_front_px']} -> "
          f"{ev_side.stats['real_front_px']}, "
          f"virtual_front {ev.stats['virtual_front_px']} -> "
          f"{ev_side.stats['virtual_front_px']}")

    # ---- 6. 缺测被显式上报，不隐藏 ----
    check("缺测像素计入统计与告警",
          ev.stats["missing_px"] > 0
          and any("缺测" in w for w in ev.warnings),
          f"missing={ev.stats['missing_px']}px")

    # ---- 7. 边缘诊断可运行且指标有限 ----
    check("边缘诊断指标有限",
          np.isfinite(ev.edge["mean_dist_px"])
          and 0.0 <= ev.edge["match_ratio"] <= 1.0,
          f"mean={ev.edge['mean_dist_px']:.2f}px "
          f"match={ev.edge['match_ratio']:.1%}")

    # ---- 7b. 人为扰动深度-颜色外参，边缘诊断必须能发现配准偏差 ----
    dcam0 = depth_camera_from_poses(f0.color_extrinsics, f0.depth_extrinsics,
                                    f0.depth_K)
    # 让深度相机光心横向偏移 60mm（模拟标定错误）
    bad_depth_ext = CameraExtrinsics(
        R=f0.depth_extrinsics.R.copy(),
        t=f0.depth_extrinsics.t + np.array([60.0, 0.0, 0.0]))
    dcam_bad = depth_camera_from_poses(f0.color_extrinsics, bad_depth_ext,
                                       f0.depth_K)
    d_mm = decode_depth(f0.depth_raw, dcam0.depth_scale, dcam0.invalid_value)
    reg_bad, valid_bad = register_depth_to_color(
        d_mm, f0.depth_K, dcam_bad, f0.color_K)
    diag_bad = diagnose_registration(f0.real_only, reg_bad, valid_bad)
    check("外参扰动被边缘诊断捕获",
          diag_bad.as_dict()["match_ratio"] < ev.edge["match_ratio"]
          and diag_bad.as_dict()["mean_dist_px"] > ev.edge["mean_dist_px"] + 1.0,
          f"match {ev.edge['match_ratio']:.1%} -> "
          f"{diag_bad.as_dict()['match_ratio']:.1%}, "
          f"mean {ev.edge['mean_dist_px']:.2f} -> "
          f"{diag_bad.as_dict()['mean_dist_px']:.2f}px")

    # ---- 8. 内核拒绝越界位姿，错误可回传 ----
    bad = np.array([MOVE_LIMITS["x"][1] + 100.0, 0.0, 0.0])
    try:
        evaluate_frame(f0, bad)
        check("越界位姿被拒绝", False, "未抛异常")
    except PipelineError as exc:
        check("越界位姿被拒绝", "超出内核允许范围" in str(exc), str(exc)[:60])

    # ---- 9. 多帧轨迹：每帧都能完成合成且关系标签合法 ----
    ok_all = True
    for fr in frames:
        e = evaluate_frame(fr, np.zeros(3))
        if e.composite_rgb.shape != (fr.color_K.height, fr.color_K.width, 3):
            ok_all = False
        if set(np.unique(e.relation_codes)) - set(RELATION_CODE.values()):
            ok_all = False
    check("移动相机轨迹全部帧合成", ok_all, f"{len(frames)} frames")

    width = max(len(n) for _, n, _ in results)
    failed = 0
    for status, name, detail in results:
        if status == FAIL:
            failed += 1
        print(f"[{status}] {name.ljust(width)}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
