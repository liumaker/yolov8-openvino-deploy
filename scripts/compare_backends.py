"""同一个输入张量，分别走 PyTorch / ONNX Runtime / OpenVINO IR 三条推理路径并计时。

用法：python scripts/compare_backends.py [--iters 20]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import numpy as np  # noqa: E402

from src.ov_engine import OVEngine  # noqa: E402
from verify_onnx import build_input, torch_raw  # noqa: E402

IMGSZ = 640


def timeit(fn, iters: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"mean": statistics.mean(times), "p50": statistics.median(times), "min": min(times)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--image", default=str(ROOT / "data" / "samples" / "bus.jpg"))
    args = ap.parse_args()

    x = build_input(Path(args.image), IMGSZ)
    print(f"输入张量 {x.shape} {x.dtype}；每种后端 warmup 3 次 + 计时 {args.iters} 次\n")

    # 1) PyTorch：逐层 Python 调度（模型只加载一次，计时只算前向）
    import contextlib
    import io

    import torch
    from ultralytics import YOLO

    with contextlib.redirect_stdout(io.StringIO()):
        model = YOLO(str(ROOT / "models" / "yolov8n.pt"), verbose=False)
        model.model.fuse()
        model.model.model[-1].export = True
        model.model.eval()

    def pt_forward():
        with torch.no_grad():
            return model.model(torch.from_numpy(x))

    t = timeit(pt_forward, args.iters)
    pt_out = pt_forward().numpy()
    print(f"PyTorch  : mean {t['mean']:.2f} ms  p50 {t['p50']:.2f}  min {t['min']:.2f}   out{pt_out.shape}")

    # 2) ONNX Runtime：session.run 触发整图执行
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx"),
                                providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    o_name = sess.get_outputs()[0].name
    t = timeit(lambda: sess.run([o_name], {i_name: x}), args.iters)
    ort_out = sess.run([o_name], {i_name: x})[0]
    print(f"ONNX RT  : mean {t['mean']:.2f} ms  p50 {t['p50']:.2f}  min {t['min']:.2f}   out{ort_out.shape}")

    # 3) OpenVINO IR：编译后直接调用 CompiledModel
    outs = {}
    for name in ("yolov8n_640_b1_fp32.xml", "yolov8n_640_b1_int8_aa2.xml"):
        eng = OVEngine(ROOT / "models" / "ir" / name)
        t = timeit(lambda: eng.infer(x), args.iters)
        outs[name] = eng.infer(x)
        print(f"OV IR {name.replace('yolov8n_640_b1_', '').replace('.xml', ''):<9}: "
              f"mean {t['mean']:.2f} ms  p50 {t['p50']:.2f}  min {t['min']:.2f}   "
              f"out{outs[name].shape}  compile {eng.compile_ms:.0f} ms")

    fp32_out = outs["yolov8n_640_b1_fp32.xml"]
    int8_out = outs["yolov8n_640_b1_int8_aa2.xml"]
    print(f"\n[一致性] max|PyTorch - ORT    | = {np.abs(pt_out - ort_out).max():.5f}")
    print(f"[一致性] max|PyTorch - IR fp32| = {np.abs(pt_out - fp32_out).max():.5f}")
    print(f"[一致性] max|IR fp32 - INT8   | = {np.abs(fp32_out - int8_out).max():.5f}")


if __name__ == "__main__":
    main()
