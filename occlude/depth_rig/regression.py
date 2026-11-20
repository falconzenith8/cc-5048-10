"""深度测试相机回归套件（纯标准库断言，同时兼容 pytest）。

四个故障在测试帧上必须落在互不相同的失败区域：

- unit_error        仅远表面（深度 >= 1200mm）且当前可见的像素；
- temporal_mismatch 仅板外轮廓/孔轮廓在相邻两帧间扫过的边缘条带，
                     与孔内区域、无效区互斥，静止薄板时必须消失；
- hole_fill         仅孔内可见区；
- missing_as_occluder 仅有效位=False 的缺测像素，且紧邻真实近表面。

运行：python3 -m occlude.depth_rig.regression（退出码 0 = 全通过）
"""
from __future__ import annotations

from dataclasses import dataclass

from depth_rig.compositor import EDGE_STATUS, MISSING_INVALID
from depth_rig.geometry import EDGE_UNCERTAIN, HOLE_VISIBLE
from depth_rig.harness import (
    RegressionHarness,
    ReplayInterface,
    buffer_diff_mask,
    count_mask,
    get_preset,
    masks_disjoint,
)
from depth_rig.scenario import PLATE_Z_MM, build_scenario
from depth_rig.sensor import UNIT_ERROR_MIN_DEPTH_MM, simulate_camera

TEST_FRAME = 12


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _analytic_masks(harness: RegressionHarness, frame_index: int) -> dict:
    """由独立射线几何（而非传感器输出）推导的各故障允许失败区域。

    深度流固定延后一帧，因此单位错误的允许区按深度来源帧 t-1 的几何
    推导；孔填充与缺测作用于当前帧，按当前帧几何推导。
    """
    scenario = harness.scenario
    frames = simulate_camera(scenario)
    now = frames[frame_index]
    src_index = max(0, frame_index - 1)
    depth_src = frames[src_index]
    h, w = scenario.grid.height_px, scenario.grid.width_px

    far_visible = [[False] * w for _ in range(h)]
    hole = [[False] * w for _ in range(h)]
    invalid = [[not now.valid[r][c] for c in range(w)] for r in range(h)]
    swept_edge = [[False] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            label = now.regions[r][c]
            if label == HOLE_VISIBLE:
                hole[r][c] = True
            # 单位错误区：当前有效，深度来源帧（t-1）该射线为板外自由
            # 射线，且来源帧前后两侧（t-2 与 t）标签都稳定（非扫动条带）、
            # 真实深度达远表面阈值。排除边缘带、孔像素与时间错配扫动像素。
            if (now.valid[r][c]
                    and depth_src.regions[r][c] == "free"
                    and frames[max(0, src_index - 1)].regions[r][c]
                    == depth_src.regions[r][c]
                    and now.regions[r][c] == depth_src.regions[r][c]
                    and depth_src.true_depth_mm[r][c] >= UNIT_ERROR_MIN_DEPTH_MM):
                far_visible[r][c] = True
            # 时间错配区：相邻两帧区域标签发生变化，且两帧都不是无效像素
            if (now.regions[r][c] != depth_src.regions[r][c]
                    and now.valid[r][c] and depth_src.valid[r][c]):
                swept_edge[r][c] = True

    return {
        "unit_error": far_visible,
        "temporal_mismatch": swept_edge,
        "hole_fill": hole,
        "missing_as_occluder": invalid,
    }


def _mask_subset(small: list[list[bool]], big: list[list[bool]]) -> bool:
    return all(not small[r][c] or big[r][c]
               for r in range(len(small)) for c in range(len(small[0])))


def run_suite() -> list[CheckResult]:
    harness = RegressionHarness()
    allowed = _analytic_masks(harness, TEST_FRAME)
    results: list[CheckResult] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(CheckResult(name, bool(ok), detail))

    # ---- 0. golden 自检：正确实现自比无失败像素 ----
    golden = harness.run_preset("golden")
    check("golden 自比零失败像素",
          count_mask(buffer_diff_mask(golden[TEST_FRAME],
                                      golden[TEST_FRAME])) == 0)
    check("golden 缺测像素保留 MISSING_INVALID 状态",
          count_mask(golden[TEST_FRAME].status_mask(MISSING_INVALID)) > 0,
          f"{count_mask(golden[TEST_FRAME].status_mask(MISSING_INVALID))} px")

    # 边缘不确定带优先级最高：落在边缘带的像素一律 EDGE_STATUS，即使它
    # 同时位于声明无效区，也不会被归入缺测。
    edge_regions = simulate_camera(harness.scenario)[TEST_FRAME].regions
    edge_priority = all(
        golden[TEST_FRAME].status[r][c] == EDGE_STATUS
        for r in range(harness.scenario.grid.height_px)
        for c in range(harness.scenario.grid.width_px)
        if edge_regions[r][c] == EDGE_UNCERTAIN)
    check("边缘不确定带优先于缺测，状态互斥", edge_priority,
          f"{sum(cell == EDGE_UNCERTAIN for row in edge_regions for cell in row)} 边缘像素")

    # ---- 1~4. 每个故障的失败像素必须非空且被限制在声明区域内 ----
    failure_masks: dict[str, list[list[bool]]] = {}
    for preset_name in ("unit_error", "temporal_mismatch",
                        "hole_fill", "missing_as_occluder"):
        mask = harness.failure_mask(preset_name, TEST_FRAME)
        failure_masks[preset_name] = mask
        n = count_mask(mask)
        inside = _mask_subset(mask, allowed[preset_name])
        check(f"{preset_name} 故障确实发生", n > 0, f"{n} 失败像素")
        check(f"{preset_name} 失败区限于声明区域", inside,
              f"{n} 失败像素，允许区 {count_mask(allowed[preset_name])} px")

    # ---- 5. 三种注入故障 + 缺测故障的失败区两两不相交 ----
    names = list(failure_masks)
    disjoint = True
    detail = ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not masks_disjoint(failure_masks[names[i]],
                                  failure_masks[names[j]]):
                disjoint = False
                detail = f"{names[i]} ∩ {names[j]} 非空"
    check("四种故障失败区两两互斥", disjoint, detail)

    # ---- 6. 单位错误只影响远表面：板/近/中层深度逐像素保持正确 ----
    # 深度来自 t-1 帧，因此「近处未被放大」要按来源帧几何判断：来源帧
    # 真实深度小于阈值的像素，输出深度必须与来源帧真实深度完全相等。
    scenario = harness.scenario
    frames = simulate_camera(scenario)
    unit_preset = get_preset("unit_error")
    unit_obs = simulate_camera(scenario, unit_preset.sensor)[TEST_FRAME]
    src_geom = frames[max(0, TEST_FRAME - 1)]
    near_ok = True
    for r in range(scenario.grid.height_px):
        for c in range(scenario.grid.width_px):
            td = src_geom.true_depth_mm[r][c]
            if (unit_obs.valid[r][c] and td < UNIT_ERROR_MIN_DEPTH_MM
                    and abs(unit_obs.depth_mm[r][c] - td) > 1e-9):
                near_ok = False
    check("单位错误不波及近处表面与薄板", near_ok)

    # ---- 7. 时间错配随速度变化：拖慢薄板后失败像素下降，静止时为 0 ----
    # 不同速度下板到达同一位姿的帧号不同，先匹配板位姿再比较，隔离
    # 「速度」变量；静止薄板位姿恒定，时间错配必须消失。
    def temporal_failure_count(speed: float) -> tuple[int, int]:
        scn = build_scenario(speed_mm_per_frame=speed)
        target_x = harness.scenario.plate_center_x(TEST_FRAME)
        matched = min(range(scn.n_frames),
                      key=lambda i: abs(scn.plate_center_x(i) - target_x))
        local = RegressionHarness(scn)
        gold = local.run_preset("golden")[matched]
        faulty = local.run_preset("temporal_mismatch")[matched]
        return matched, count_mask(buffer_diff_mask(gold, faulty))

    idx_fast_t, n_fast = temporal_failure_count(
        ReplayInterface.fast_speed())
    idx_slow_t, n_slow = temporal_failure_count(
        ReplayInterface.slow_speed())
    _, n_still = temporal_failure_count(0.0)
    check("时间错配随薄板速度下降而减弱", 0 < n_slow < n_fast,
          f"fast#{idx_fast_t}={n_fast} slow#{idx_slow_t}={n_slow} "
          f"still={n_still} px")
    check("薄板静止时时间错配消失", n_still == 0, f"{n_still} px")

    # ---- 8. 距离对照：板平面平移后时间错配规模不变（正交相机与距离无关）----
    golden_replay = ReplayInterface("golden")
    replay = ReplayInterface("temporal_mismatch")
    shifted = replay.replay(TEST_FRAME,
                            speed_mm_per_frame=
                            ReplayInterface.fast_speed(),
                            plate_z_mm=PLATE_Z_MM + 150.0)
    gold_shifted = golden_replay.replay(
        TEST_FRAME,
        speed_mm_per_frame=ReplayInterface.fast_speed(),
        plate_z_mm=PLATE_Z_MM + 150.0)
    n_shift = count_mask(buffer_diff_mask(gold_shifted["target_buffer"],
                                          shifted["target_buffer"]))
    check("时间错配由速度而非板距离驱动", n_shift == n_fast,
          f"z={PLATE_Z_MM}: {n_fast} px -> z={PLATE_Z_MM + 150}: {n_shift} px")

    # ---- 9. 单位错误随深度（距离）变化、与薄板速度无关 ----
    # 单位错误是传感器对「稳定远自由射线」的固定比例换算：取慢速想定中
    # 与测试帧板位姿相同的帧，逐像素验证换算规则只由射线几何决定，
    # 不随板速度变化；被换算像素必须全部落在稳定远自由射线上。
    scn_slow = build_scenario(speed_mm_per_frame=
                              ReplayInterface.slow_speed())
    target_x = harness.scenario.plate_center_x(TEST_FRAME)
    matched_slow = min(range(scn_slow.n_frames),
                       key=lambda i: abs(scn_slow.plate_center_x(i) - target_x))
    unit_preset = get_preset("unit_error")
    faulty_slow = simulate_camera(scn_slow, unit_preset.sensor)[matched_slow]
    base_slow = simulate_camera(scn_slow)[matched_slow]
    slow_frames = simulate_camera(scn_slow)
    src_slow = max(0, matched_slow - 1)
    speed_independent = True
    n_scaled_slow = 0
    for r in range(scn_slow.grid.height_px):
        for c in range(scn_slow.grid.width_px):
            if not faulty_slow.valid[r][c]:
                continue
            stable_far = (
                slow_frames[src_slow].regions[r][c] == "free"
                and slow_frames[max(0, src_slow - 1)].regions[r][c] == "free"
                and slow_frames[matched_slow].regions[r][c] == "free"
                and slow_frames[src_slow].true_depth_mm[r][c]
                >= UNIT_ERROR_MIN_DEPTH_MM)
            scaled = abs(faulty_slow.depth_mm[r][c]
                         - base_slow.depth_mm[r][c]) > 1e-9
            if scaled:
                n_scaled_slow += 1
            if scaled != stable_far:
                speed_independent = False
            if scaled:
                expect = (slow_frames[src_slow].true_depth_mm[r][c]
                          * unit_preset.sensor.unit_error_scale)
                if abs(faulty_slow.depth_mm[r][c] - expect) > 1e-9:
                    speed_independent = False
    check("单位错误与薄板速度无关（仅由稳定远射线几何决定）",
          speed_independent and n_scaled_slow > 0,
          f"慢速匹配帧 #{matched_slow}，换算像素 {n_scaled_slow}")

    # ---- 10. 缺测规则：零深度无效区与近表面相邻，且不与时间错配混淆 ----
    miss = failure_masks["missing_as_occluder"]
    miss_allowed = allowed["missing_as_occluder"]
    check("缺测故障只命中有效位=False 的像素",
          _mask_subset(miss, miss_allowed),
          f"{count_mask(miss)} px")
    check("缺测失败区与时间错配失败区互斥",
          masks_disjoint(miss, failure_masks["temporal_mismatch"]))
    # 相邻性：无效矩形 (y 54..72) 与近表面 (y 30..52) 之间只隔 2mm
    scenario0 = build_scenario()
    invalid_patch = next(p for p in scenario0.invalid_patches
                         if p.name == "invalid_zero_patch")
    near = next(s for s in scenario0.surfaces
                if s.name == "near_adjacent")
    gap = invalid_patch.y0 - near.y1
    check("零深度无效区紧邻真实近表面", 0.0 < gap <= 5.0, f"间隙 {gap}mm")

    # golden 必须把无效区判为缺测而不是最近遮挡：深度保持 0、状态 MISSING
    buf = golden[TEST_FRAME]
    invalid_depth_zero = all(
        buf.depth_mm[r][c] == 0.0
        for r in range(buf.height_px) for c in range(buf.width_px)
        if not frames[TEST_FRAME].valid[r][c])
    check("golden 对缺测保留零深度并显式标记，不当最近遮挡",
          invalid_depth_zero)

    # ---- 11. 回放携带前序时间戳，且返回的就是最终显示缓冲 ----
    rep = replay.replay(TEST_FRAME, history=2)
    check("回放返回失败帧及前序时间戳",
          rep["history_indices"] == [TEST_FRAME - 2, TEST_FRAME - 1,
                                     TEST_FRAME]
          and len(rep["history_timestamps_s"]) == 3,
          str(rep["history_timestamps_s"]))
    check("回放目标是 DisplayBuffer 最终显示缓冲",
          type(rep["target_buffer"]).__name__ == "DisplayBuffer"
          and rep["target_buffer"].frame_index == TEST_FRAME)

    return results


def main() -> int:
    results = run_suite()
    width = max(len(r.name) for r in results)
    failed = 0
    for r in results:
        if not r.passed:
            failed += 1
        print(f"[{'PASS' if r.passed else 'FAIL'}] "
              f"{r.name.ljust(width)}  {r.detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
