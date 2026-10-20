#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 2 步（验收）：IR 与 ONNX 的数值等价性。

做法：同一张图 -> 同一套 letterbox -> 同一个 float32 张量，分别喂
    ONNX(onnxruntime) / IR(fp32) / IR(fp16)
比较原始输出张量（还没做 NMS，所以比的是纯网络输出）。

预期
    fp32 IR 与 ONNX：max|Δ| ~1e-5 量级（浮点求和顺序变了而已）
    fp16 IR 与 ONNX：max|Δ| ~1e-3 量级（权重被截断成 fp16 后再转回 fp32 计算）
如果 fp32 都对不上，说明转换/前处理出了问题，不用再往下走。

用法
    python scripts/verify_ir.py
    python scripts/verify_ir.py --onnx models/onnx/yolov8n_640_b1.onnx --irdir models/ir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_onnx import build_input  # 复用第 1 步写好的 letterbox 前处理

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IR vs ONNX 数值对比")
    p.add_argument("--onnx", type=Path, default=ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx")
    p.add_argument("--irdir", type=Path, default=ROOT / "models" / "ir")
    p.add_argument("--prefix", default="yolov8n_640_b1")
    p.add_argument("--image", type=Path, default=ROOT / "data" / "samples" / "bus.jpg")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="CPU")
    return p.parse_args()


def ort_output(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    so = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return so.run(None, {so.get_inputs()[0].name: x})[0]


def ov_output(xml: Path, x: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    import openvino as ov
    core = ov.Core()
    t0 = time.perf_counter()
    compiled = core.compile_model(str(xml), device)
    compile_s = time.perf_counter() - t0
    return np.asarray(compiled([x])[compiled.output(0)]), compile_s


def diff_stats(ref: np.ndarray, got: np.ndarray) -> dict:
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    scale = max(float(np.abs(ref).max()), 1e-9)
    # 前 4 行是框坐标(已乘 stride，量级到几百)，后面是 80 类 logits(量级个位数)
    box_part = d[:, :4, :].max() if d.ndim == 3 and d.shape[1] > 4 else d.max()
    cls_part = d[:, 4:, :].max() if d.ndim == 3 and d.shape[1] > 4 else d.max()
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rel_max": float(d.max() / scale),
        "max_abs_box": float(box_part),
        "max_abs_cls": float(cls_part),
        "shape_ok": ref.shape == got.shape,
    }


def main() -> int:
    args = parse_args()
    x = build_input(args.image.resolve(), args.imgsz)
    print(f"输入张量 {x.shape} {x.dtype}  min={x.min():.3f} max={x.max():.3f}")

    t0 = time.perf_counter()
    ref = ort_output(args.onnx.resolve(), x)
    print(f"ONNX 输出 {ref.shape}  (onnxruntime {time.perf_counter() - t0:.2f}s)  "
          f"absmax={np.abs(ref).max():.3f}")

    report = {"onnx": str(args.onnx), "image": str(args.image), "results": []}
    ok = True
    for prec in ["fp32", "fp16"]:
        xml = args.irdir / f"{args.prefix}_{prec}.xml"
        if not xml.is_file():
            print(f"[skip] {xml.name} 不存在")
            continue
        got, compile_s = ov_output(xml, x, args.device)
        st = diff_stats(ref, got)
        # 用**相对**阈值：输出里框坐标量级可达 600+，绝对阈值没有通用意义
        thresh = 1e-4 if prec == "fp32" else 2e-2
        passed = st["shape_ok"] and st["rel_max"] <= thresh
        ok &= passed
        print(f"IR {prec}: 输出 {got.shape}  compile {compile_s:.2f}s  "
              f"max|Δ|={st['max_abs']:.3e} (框 {st['max_abs_box']:.2e} / logits {st['max_abs_cls']:.2e})  "
              f"相对={st['rel_max']:.2e}  -> {'PASS' if passed else 'FAIL'} (阈值 相对 {thresh:g})")
        report["results"].append({"precision": prec, "compile_s": round(compile_s, 3), **st,
                                  "pass": passed})

    out = args.irdir / f"{args.prefix}.verify.json"
    report["pass"] = ok
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'}  报告写入 {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
