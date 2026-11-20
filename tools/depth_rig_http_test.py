#!/usr/bin/env python3
"""深度测试相机 HTTP 层集成测试（纯标准库）。

在临时端口启动真实回放服务，通过 HTTP 取回最终显示缓冲，逐像素比对
内核直接产出的 DisplayBuffer，确保界面读到的数据与回归断言同源；
并校验失败区、速度/距离回放与缺测规则在 API 层同样成立。

用法：python3 tools/depth_rig_http_test.py   退出码 0 = 全部通过。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "occlude"))

from depth_rig.compositor import MISSING_INVALID  # noqa: E402
from depth_rig.harness import RegressionHarness  # noqa: E402
from depth_rig.regression import TEST_FRAME, run_suite  # noqa: E402
from depth_rig.server import make_handler  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))


def start_server() -> str:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    base = start_server()

    info = get(base, "/api/info")
    check("info 暴露全部五种预设",
          info["presets"] == ["golden", "unit_error", "temporal_mismatch",
                              "hole_fill", "missing_as_occluder"],
          str(info["presets"]))

    # 1. API 显示缓冲与内核 DisplayBuffer 逐像素一致（颜色/深度/状态）
    harness = RegressionHarness()
    golden_buf = harness.run_preset("golden")[TEST_FRAME]
    payload = get(base, f"/api/frame?preset=golden&frame={TEST_FRAME}&history=2")
    api_buf = payload["buffer"]
    same_color = api_buf["color"] == [[list(p) for p in row]
                                      for row in golden_buf.color]
    same_depth = api_buf["depth_mm"] == [list(r) for r in golden_buf.depth_mm]
    same_status = api_buf["status"] == [list(r) for r in golden_buf.status]
    check("API 颜色缓冲 == 内核最终显示缓冲", same_color)
    check("API 深度缓冲 == 内核最终显示缓冲", same_depth)
    check("API 状态缓冲 == 内核最终显示缓冲", same_status)
    check("回放携带失败帧及前序时间戳",
          payload["history_indices"] == [TEST_FRAME - 2, TEST_FRAME - 1,
                                         TEST_FRAME],
          str(payload["history_timestamps_s"]))

    # 2. 每个故障的 API 失败区与线束失败区逐像素相同且非空
    for preset in ("unit_error", "temporal_mismatch", "hole_fill",
                   "missing_as_occluder"):
        d = get(base, f"/api/diff?preset={preset}&frame={TEST_FRAME}")
        local_mask = harness.failure_mask(preset, TEST_FRAME)
        check(f"/api/diff {preset} 与线束失败区一致",
              d["failure_mask"] == local_mask,
              f"{d['failure_count']} px")
        check(f"/api/diff {preset} 失败非空", d["failure_count"] > 0)

    # 3. 缺测规则在 API 层成立：非边缘带的无效像素为 missing_invalid、深度 0。
    # 边缘不确定带优先级最高，落在边缘带内的无效像素归 edge_uncertain。
    from depth_rig.geometry import EDGE_UNCERTAIN
    from depth_rig.sensor import simulate_camera
    frames = simulate_camera(harness.scenario)

    def is_pure_invalid(r: int, c: int) -> bool:
        return (not frames[TEST_FRAME].valid[r][c]
                and frames[TEST_FRAME].regions[r][c] != EDGE_UNCERTAIN)

    invalid_status_ok = all(
        api_buf["status"][r][c] == MISSING_INVALID
        and api_buf["depth_mm"][r][c] == 0.0
        for r in range(api_buf["height_px"])
        for c in range(api_buf["width_px"])
        if is_pure_invalid(r, c))
    check("API 缺测像素保持零深度并标记 missing_invalid", invalid_status_ok)

    # missing_as_occluder 把同一批纯缺测像素改成板色/板深度（被检出）
    miss_payload = get(base,
                       f"/api/frame?preset=missing_as_occluder&frame={TEST_FRAME}")
    wrong = [
        (r, c) for r in range(api_buf["height_px"])
        for c in range(api_buf["width_px"])
        if is_pure_invalid(r, c)
        and (miss_payload["buffer"]["status"][r][c] != "ok_valid"
             or miss_payload["buffer"]["depth_mm"][r][c]
             != harness.scenario.plate.z_plate_mm)
    ]
    check("API 可检出缺测被当最近遮挡（状态/深度被改写）", wrong == [])

    # 4. 速度/距离回放：拖慢后时间错配下降；板距离平移不改变其规模
    fast = get(base, "/api/diff?preset=temporal_mismatch&frame=12&speed=18")
    slow = get(base, "/api/diff?preset=temporal_mismatch&frame=12&speed=3.75")
    check("API 拖慢薄板使时间错配失败下降",
          0 < slow["failure_count"] < fast["failure_count"],
          f"fast={fast['failure_count']} slow={slow['failure_count']}")
    near = get(base, "/api/frame?preset=temporal_mismatch&frame=12&plate_z=400")
    far = get(base, "/api/frame?preset=temporal_mismatch&frame=12&plate_z=550")
    check("API 板距离改变但时间错配规模不变（随速度不随距离）",
          near["buffer"]["stats"]["temporal_overwritten_px"]
          == far["buffer"]["stats"]["temporal_overwritten_px"],
          str(near["buffer"]["stats"]["temporal_overwritten_px"]))

    # 5. HTTP 触发整套回归，结果与直接运行一致且全通过
    reg = get(base, "/api/regression")
    local_pass = sum(1 for r in run_suite() if r.passed)
    check("HTTP 回归结果与本地一致且全通过",
          reg["passed"] == reg["total"] == local_pass,
          f"{reg['passed']}/{reg['total']}")

    width = max(len(n) for _, n, _ in results)
    failed = 0
    for status, name, detail in results:
        if status == FAIL:
            failed += 1
        print(f"[{status}] {name.ljust(width)}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} http checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
