#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 2 步：ONNX -> OpenVINO IR（.xml 图 + .bin 权重）。

ovc / ov.convert_model 在转换期做的事
1. ONNX 前端解析：把 ONNX 图翻译成 OpenVINO 的 nGraph 表示（算子名会变，如 Clip -> Clamp、
   Resize -> Interpolate）；
2. 图变换：常量折叠、算子融合（Conv+BN+ReLU 合成一个节点）、冗余 Reshape 消除；
3. 形状/布局推断：静态模型在这里把维度全部算实；
4. 序列化：xml 存结构，bin 存权重。

注意 compress_to_fp16 只压缩**权重**，计算仍是 fp32 —— 这是 CPU 上说的 "FP16 IR"，
省的是磁盘和内存，不是算力。真正的 FP16 计算要 GPU 才受益。

用法
    python scripts/export_ir.py                       # 静态 ONNX -> fp32 + fp16
    python scripts/export_ir.py --precision fp32      # 只转一份
    python scripts/export_ir.py --onnx models/onnx/yolov8n_640_dyn.onnx --name yolov8n_640_dyn
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX -> OpenVINO IR")
    p.add_argument("--onnx", type=Path, default=ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx")
    p.add_argument("--outdir", type=Path, default=ROOT / "models" / "ir")
    p.add_argument("--name", default=None, help="输出文件名前缀，默认取 onnx 文件名")
    p.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    return p.parse_args()


def describe(model, tag: str, xml: Path) -> dict:
    inp = model.inputs[0]
    out = model.outputs[0]
    ops = Counter(op.get_type_name() for op in model.get_ordered_ops())
    n_ops = sum(ops.values())
    n_real = n_ops - ops.get("Constant", 0) - ops.get("Parameter", 0) - ops.get("Result", 0)
    bin_path = xml.with_suffix(".bin")

    print(f"  [{tag}] 输入   : {inp.get_partial_shape()}  {inp.get_element_type()}")
    print(f"  [{tag}] 输出   : {out.get_partial_shape()}  {out.get_element_type()}")
    print(f"  [{tag}] 节点数 : {n_ops}（计经常量/输入输出后 {n_real}）   Top8: {ops.most_common(8)}")
    print(f"  [{tag}] 文件   : {xml.name} {xml.stat().st_size / 1e3:.1f} KB + "
          f"{bin_path.name} {bin_path.stat().st_size / 1e6:.2f} MB")

    return {
        "tag": tag,
        "input_shape": str(inp.get_partial_shape()),
        "input_type": str(inp.get_element_type()),
        "output_shape": str(out.get_partial_shape()),
        "num_ops": n_ops,
        "num_ops_real": n_real,
        "op_counts": dict(ops),
        "xml_kb": round(xml.stat().st_size / 1e3, 1),
        "bin_mb": round(bin_path.stat().st_size / 1e6, 2),
    }


def main() -> int:
    args = parse_args()
    import onnx
    import openvino as ov

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"[x] 找不到 ONNX: {onnx_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.name or onnx_path.stem
    targets = ["fp32", "fp16"] if args.precision == "both" else [args.precision]

    n_onnx_nodes = len(onnx.load(str(onnx_path)).graph.node)
    print(f"源 ONNX: {onnx_path.name}  节点 {n_onnx_nodes} 个")
    print(f"OpenVINO: {ov.get_version()}\n")

    info = []
    for prec in targets:
        xml = args.outdir / f"{prefix}_{prec}.xml"
        t0 = time.perf_counter()
        model = ov.convert_model(str(onnx_path))
        t_parse = time.perf_counter() - t0
        ov.save_model(model, str(xml), compress_to_fp16=(prec == "fp16"))
        cost = time.perf_counter() - t0

        print(f"--- {prec}  解析 {t_parse:.2f}s / 转换+序列化 {cost:.2f}s ---")
        m = ov.Core().read_model(str(xml))
        info.append(describe(m, prec, xml))

    meta = {"source_onnx": str(onnx_path), "onnx_nodes": n_onnx_nodes,
            "openvino": ov.get_version(), "precision": info}
    meta_path = args.outdir / f"{prefix}.ir.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n元信息写入 {meta_path}")
    if info:
        first = info[0]
        print(f"ONNX {n_onnx_nodes} 节点 -> IR {first['num_ops']} 节点"
              f"（其中 Constant {first['op_counts'].get('Constant', 0)} 个，非 Constant {first['num_ops_real']} 个；"
              f"ONNX 的 initializer 在 IR 里会变成 Constant 节点，所以拿来直接比节点总数没有意义）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
