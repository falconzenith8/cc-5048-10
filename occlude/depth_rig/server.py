"""深度测试相机回放 HTTP 服务（纯标准库）。

启动：python3 -m depth_rig.server [--port 8100]
（或 python3 tools/depth_rig_server.py）

界面与回归断言读取同一份数据：所有视图都由 DisplayBuffer（最终显示
缓冲）序列化得到，服务不另造一套渲染结果。API：

- GET /api/info                       想定参数、预设、速度选项
- GET /api/frame?preset=&frame=&speed=&plate_z=&history=
      返回目标帧与前序时间戳的完整显示缓冲（颜色/深度/状态/统计）
- GET /api/regression                 运行回归套件并回传逐条结果
- GET /api/diff?preset=&frame=&speed= 该预设相对 golden 的逐像素失败区
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from depth_rig.compositor import STATUS_CODES
from depth_rig.geometry import REGIONS
from depth_rig.harness import (
    PRESETS,
    RegressionHarness,
    ReplayInterface,
    buffer_diff_mask,
    count_mask,
    get_preset,
)
from depth_rig.regression import run_suite
from depth_rig.scenario import (
    PLATE_SPEED_FAST_MM_PER_FRAME,
    PLATE_SPEED_SLOW_MM_PER_FRAME,
    PLATE_Z_MM,
    build_scenario,
)
from depth_rig.sensor import simulate_camera

WEB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "depth_rig_web")


def _buffer_payload(buf) -> dict:
    """把最终显示缓冲序列化为 JSON 友好的逐像素结构。"""
    return {
        "frame_index": buf.frame_index,
        "timestamp_s": buf.timestamp_s,
        "width_px": buf.width_px,
        "height_px": buf.height_px,
        "status_codes": STATUS_CODES,
        "color": [[list(px) for px in row] for row in buf.color],
        "depth_mm": [list(row) for row in buf.depth_mm],
        "status": [list(row) for row in buf.status],
        "stats": buf.stats,
    }


def _frame_payload(preset: str, frame: int, speed: float,
                   plate_z: float | None, history: int) -> dict:
    replay = ReplayInterface(preset)
    rep = replay.replay(frame, speed_mm_per_frame=speed,
                        plate_z_mm=plate_z, history=history)
    scenario = build_scenario(speed_mm_per_frame=speed,
                              plate_z_mm=rep["plate_z_mm"])
    obs = simulate_camera(scenario, get_preset(preset).sensor)[frame]
    return {
        "preset": preset,
        "frame_index": rep["frame_index"],
        "speed_mm_per_frame": rep["speed_mm_per_frame"],
        "plate_z_mm": rep["plate_z_mm"],
        "history_indices": rep["history_indices"],
        "history_timestamps_s": rep["history_timestamps_s"],
        "plate_center_x_mm": obs.plate_center_x_mm,
        "regions": obs.regions,
        "region_names": list(REGIONS),
        "buffer": _buffer_payload(rep["target_buffer"]),
        "history_buffers": [_buffer_payload(b) for b in rep["buffers"]],
    }


def _diff_payload(preset: str, frame: int, speed: float) -> dict:
    scenario = build_scenario(speed_mm_per_frame=speed)
    harness = RegressionHarness(scenario)
    # 速度变化后板位姿对应的帧号可能改变，先匹配与默认想定相同的板位姿
    target_x = build_scenario().plate_center_x(frame)
    matched = min(range(scenario.n_frames),
                  key=lambda i: abs(scenario.plate_center_x(i) - target_x))
    golden = harness.run_preset("golden")[matched]
    faulty = harness.run_preset(preset)[matched]
    mask = buffer_diff_mask(golden, faulty)
    return {
        "preset": preset,
        "frame_index": matched,
        "failure_count": count_mask(mask),
        "failure_mask": mask,
        "golden": _buffer_payload(golden),
        "faulty": _buffer_payload(faulty),
    }


def make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "DepthRigServer/1.0"

        def log_message(self, fmt, *args):
            print(f"[depth-rig] {self.address_string()} {fmt % args}")

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, rel: str, ctype: str) -> None:
            path = os.path.normpath(os.path.join(WEB_DIR, rel))
            if not path.startswith(WEB_DIR) or not os.path.isfile(path):
                self._send_json(404, {"error": f"资源不存在: {rel}"})
                return
            with open(path, "rb") as fp:
                body = fp.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path in ("/", "/index.html"):
                    return self._send_file("index.html",
                                           "text/html; charset=utf-8")
                if parsed.path == "/app.js":
                    return self._send_file("app.js",
                                           "application/javascript; charset=utf-8")
                if parsed.path == "/style.css":
                    return self._send_file("style.css", "text/css; charset=utf-8")

                if parsed.path == "/api/info":
                    scn = build_scenario()
                    return self._send_json(200, {
                        "presets": list(PRESETS),
                        "frame_count": scn.n_frames,
                        "frame_period_s": scn.frame_period_s,
                        "grid": {
                            "width_px": scn.grid.width_px,
                            "height_px": scn.grid.height_px,
                            "view_w_mm": scn.grid.view_w_mm,
                            "view_h_mm": scn.grid.view_h_mm,
                            "pixel_size_mm": scn.grid.pixel_size_mm,
                        },
                        "speeds_mm_per_frame": {
                            "fast": PLATE_SPEED_FAST_MM_PER_FRAME,
                            "slow": PLATE_SPEED_SLOW_MM_PER_FRAME,
                            "still": 0.0,
                        },
                        "default_plate_z_mm": PLATE_Z_MM,
                        "status_codes": STATUS_CODES,
                        "regions": list(REGIONS),
                    })

                def num(name: str, default: float) -> float:
                    try:
                        return float(qs.get(name, [str(default)])[0])
                    except ValueError:
                        raise ValueError(f"参数 {name} 必须是数字")

                if parsed.path == "/api/frame":
                    preset = qs.get("preset", ["golden"])[0]
                    if preset not in PRESETS:
                        raise ValueError(f"未知预设 {preset}")
                    frame = int(num("frame", 12))
                    speed = num("speed", PLATE_SPEED_FAST_MM_PER_FRAME)
                    plate_z = qs.get("plate_z", [None])[0]
                    plate_z = float(plate_z) if plate_z is not None else None
                    history = int(num("history", 1))
                    return self._send_json(200, _frame_payload(
                        preset, frame, speed, plate_z, history))

                if parsed.path == "/api/diff":
                    preset = qs.get("preset", ["unit_error"])[0]
                    if preset not in PRESETS or preset == "golden":
                        raise ValueError(f"未知故障预设 {preset}")
                    frame = int(num("frame", 12))
                    speed = num("speed", PLATE_SPEED_FAST_MM_PER_FRAME)
                    return self._send_json(200,
                                           _diff_payload(preset, frame, speed))

                if parsed.path == "/api/regression":
                    results = run_suite()
                    passed = sum(1 for r in results if r.passed)
                    return self._send_json(200, {
                        "checks": [{"name": r.name, "passed": r.passed,
                                    "detail": r.detail} for r in results],
                        "passed": passed,
                        "total": len(results),
                    })

                self._send_json(404, {"error": f"未知路径 {parsed.path}"})
            except (ValueError, IndexError, KeyError) as exc:
                self._send_json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="深度测试相机回放服务")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"深度测试相机回放服务已启动: http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务")


if __name__ == "__main__":
    main()
