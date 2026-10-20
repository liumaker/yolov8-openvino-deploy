#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在本地起一个 Netron 网页服务，用浏览器看 ONNX / IR 的计算图。

用法
    python scripts/serve_netron.py                                  # 默认看静态 ONNX，端口 8081
    python scripts/serve_netron.py models/onnx/yolov8n_640_dyn.onnx 8082
然后浏览器打开打印出来的 http://127.0.0.1:8081
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import netron

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8081
    if not model.is_file():
        raise SystemExit(f"[x] 找不到模型: {model}")

    try:
        # netron >= 9 的签名是 start(file, address, browse)，address 必须是 (host, port) 元组
        netron.start(str(model), address=("127.0.0.1", port), browse=False)
    except TypeError:  # 老版本：start(file, host, port, browse)
        netron.start(str(model), port=port, browse=False)

    print(f"[netron] {model.name} -> http://127.0.0.1:{port}", flush=True)
    while True:  # 保持进程存活，Ctrl+C 退出
        time.sleep(3600)


if __name__ == "__main__":
    main()
