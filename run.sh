#!/usr/bin/env bash
# 启动实景虚拟展品合成器界面。
# 用法：./run.sh [--port 8000] [--data data/demo]
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

# 依赖缺失时给出明确提示，而不是在导入处崩溃
if ! "$PY" -c "import numpy, PIL" 2>/dev/null; then
  echo "缺少依赖：numpy / pillow。安装："
  echo "  $PY -m pip install --user numpy pillow"
  exit 1
fi

PORT=8000
DATA="data/demo"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    *) echo "未知参数 $1"; exit 2 ;;
  esac
done

echo "先运行内核自检……"
"$PY" tools/selftest.py
echo
echo "启动界面：浏览器打开 http://127.0.0.1:${PORT}/"
exec "$PY" -m occlude.server --host 127.0.0.1 --port "$PORT" --data "$DATA"
