"""实景遮挡内核的 HTTP 服务（仅用标准库，无第三方 Web 框架）。

启动：python3 -m occlude.server [--data data/demo] [--port 8000]
所有计算请求由 occlude.pipeline 内核驱动；内核拒绝（如越界位姿）以
HTTP 400 + 中文错误信息返回，前端必须回显到界面，不允许静默吞掉。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .dataset import (
    FAR_MM,
    INVALID_VALUE,
    NEAR_MM,
    REFERENCE_EXHIBIT,
    load_recording,
    simulate_recording,
    save_recording,
)
from .pipeline import (
    MOVE_LIMITS,
    PipelineError,
    RELATION_NAME,
    encode_png,
    evaluate_frame,
)

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class AppState:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        if os.path.isdir(data_dir) and os.path.exists(os.path.join(data_dir, "meta.json")):
            self.frames, self.meta = load_recording(data_dir)
        else:
            os.makedirs(data_dir, exist_ok=True)
            self.frames = simulate_recording(n_frames=12)
            save_recording(self.frames, data_dir)
            self.frames, self.meta = load_recording(data_dir)
        self._cache: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def evaluate(self, frame_index: int, ox: float, oy: float, oz: float,
                 bias: float, show_missing: bool) -> dict:
        key = (frame_index, round(ox, 2), round(oy, 2), round(oz, 2),
               round(bias, 2), show_missing)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not (0 <= frame_index < len(self.frames)):
            raise PipelineError(
                f"帧序号 {frame_index} 超出范围 [0, {len(self.frames) - 1}]。")
        ev = evaluate_frame(
            self.frames[frame_index],
            np_offset(ox, oy, oz),
            depth_bias_mm=bias,
            show_missing_warning=show_missing,
        )
        payload = {
            "frame_index": ev.frame_index,
            "exhibit_xyz_mm": [float(x) for x in ev.exhibit_xyz],
            "reference_exhibit_xyz_mm": list(REFERENCE_EXHIBIT),
            "stats": ev.stats,
            "edge": ev.edge,
            "warnings": ev.warnings,
            "move_limits": MOVE_LIMITS,
            "near_mm": NEAR_MM,
            "far_mm": FAR_MM,
            "depth_invalid_value": INVALID_VALUE,
            "relation_names": {
                str(code): name for code, name in RELATION_NAME.items()
            },
            "_ev": ev,
        }
        with self._lock:
            self._cache[key] = payload
        return payload


def np_offset(ox: float, oy: float, oz: float) -> np.ndarray:
    return np.array([ox, oy, oz], dtype=np.float64)


def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OccludeServer/1.0"

        def log_message(self, fmt, *args):  # 精简日志
            print(f"[http] {self.address_string()} {fmt % args}")

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_png(self, png: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(png)

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
                    return self._send_file("index.html", "text/html; charset=utf-8")
                if parsed.path == "/app.js":
                    return self._send_file("app.js", "application/javascript; charset=utf-8")
                if parsed.path == "/style.css":
                    return self._send_file("style.css", "text/css; charset=utf-8")
                if parsed.path == "/api/info":
                    return self._send_json(200, {
                        "frame_count": len(state.frames),
                        "meta": {k: v for k, v in state.meta.items()
                                 if k != "frames"},
                        "move_limits": MOVE_LIMITS,
                        "near_mm": NEAR_MM,
                        "far_mm": FAR_MM,
                        "depth_invalid_value": INVALID_VALUE,
                        "reference_exhibit_xyz_mm": list(REFERENCE_EXHIBIT),
                    })

                def num(name: str, default: float) -> float:
                    try:
                        return float(qs.get(name, [str(default)])[0])
                    except ValueError:
                        raise PipelineError(f"参数 {name} 必须是数字。")

                if parsed.path == "/api/gt_check":
                    fi = int(num("frame", 0))
                    if not (0 <= fi < len(state.frames)):
                        raise PipelineError(
                            f"帧序号 {fi} 超出范围 [0, {len(state.frames) - 1}]。")
                    # 真值只在参考位姿（零偏移）下有意义
                    payload = state.evaluate(fi, 0.0, 0.0, 0.0, 10.0, True)
                    ev = payload["_ev"]
                    gt = state.frames[fi].gt_composite.astype(np.int16)
                    ours = ev.composite_rgb.astype(np.int16)
                    diff = np.abs(gt - ours).mean(axis=2)
                    missing_code = next(
                        c for c, n in RELATION_NAME.items() if n == "real_missing")
                    core = ev.relation_codes != missing_code
                    return self._send_json(200, {
                        "frame_index": fi,
                        "mae_rgb": float(diff[core].mean()),
                        "mismatch_ratio": float((diff[core] > 40).mean()),
                        "compared_px": int(core.sum()),
                        "note": "缺测斜纹区与真值定义不同，已从比较中排除；"
                                "mismatch 为色差>40 的像素占比。",
                    })

                if parsed.path in ("/api/evaluate", "/api/image"):
                    fi = int(num("frame", 0))
                    ox, oy, oz = num("ox", 0), num("oy", 0), num("oz", 0)
                    bias = num("bias", 10.0)
                    show_missing = qs.get("missing", ["1"])[0] != "0"
                    payload = state.evaluate(fi, ox, oy, oz, bias, show_missing)
                    if parsed.path == "/api/evaluate":
                        public = {k: v for k, v in payload.items() if k != "_ev"}
                        return self._send_json(200, public)
                    view = qs.get("view", ["composite"])[0]
                    ev = payload["_ev"]
                    views = {
                        "composite": ev.composite_rgb,
                        "depth": ev.depth_view_rgb,
                        "mask": ev.mask_view_rgb,
                        "edge": ev.edge_overlay_rgb,
                        "gt": state.frames[fi].gt_composite,
                        "real": state.frames[fi].real_only,
                    }
                    if view not in views:
                        raise PipelineError(
                            f"未知视图 {view}，可选 {sorted(views)}。")
                    return self._send_png(encode_png(views[view]))
                self._send_json(404, {"error": f"未知路径 {parsed.path}"})
            except PipelineError as exc:
                self._send_json(400, {"error": str(exc)})
            except (ValueError, IndexError) as exc:
                self._send_json(400, {"error": f"请求参数无效：{exc}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="实景虚拟展品合成器服务")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    state = AppState(args.data)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"实景遮挡内核服务已启动: http://{args.host}:{args.port}/")
    print(f"录制数据目录: {os.path.abspath(args.data)}（{len(state.frames)} 帧）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务")


if __name__ == "__main__":
    main()
