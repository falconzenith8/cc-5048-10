#!/usr/bin/env python3
"""深度测试相机回放服务启动入口（纯标准库）。

用法：python3 tools/depth_rig_server.py [--port 8100]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "occlude"))

from depth_rig.server import main  # noqa: E402

if __name__ == "__main__":
    main()
