"""生成带真值的模拟录制（无硬件时使用）。

用法：
  python3 tools/generate_data.py --frames 12 --out data/demo
输出颜色帧、uint16 深度帧（毫米，0 为无效值）、真值合成帧与 meta.json，
格式见 README「录制格式」一节。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from occlude.dataset import save_recording, simulate_recording


def main() -> None:
    ap = argparse.ArgumentParser(description="生成带真值的模拟录制")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="data/demo")
    args = ap.parse_args()

    frames = simulate_recording(n_frames=args.frames)
    out = os.path.abspath(args.out)
    save_recording(frames, out)
    print(f"已写入 {len(frames)} 帧到 {out}")
    print("包含：color_*.png（颜色）、depth_*.png（uint16 毫米深度，0 无效）、"
          "gt_*.png（真值合成）、meta.json（双相机标定与轨迹）")


if __name__ == "__main__":
    main()
