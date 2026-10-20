#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 1 步：YOLOv8 PyTorch -> ONNX 导出。

关键点
- opset：OpenVINO 支持 11~19，这里固定 12（YOLOv8 用到的算子 12 就能表达，兼容性最好）；
- dynamic=True：batch / height / width 全部符号化，一个模型可跑任意分辨率与批大小，
  但静态 shape 在 CPU 上通常更快，所以两种都导一份做对比；
- simplify：用 onnxslim 清掉导出过程中的冗余节点（Constant folding / Identity 等）。

用法
    python scripts/export_onnx.py                     # 640, batch=1, 静态
    python scripts/export_onnx.py --dynamic           # 640, batch/宽高动态
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ultralytics 默认往 %APPDATA%\Ultralytics 写 settings.json，这里收回项目内
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultralytics YOLOv8 -> ONNX")
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640, help="导出时的方形边长")
    p.add_argument("--batch", type=int, default=1, help="静态导出时的 batch，dynamic 时忽略")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--dynamic", action="store_true", help="batch 与 H/W 设为动态轴")
    p.add_argument("--no-simplify", action="store_true", help="跳过 onnxslim 图简化")
    p.add_argument("--outdir", type=Path, default=ROOT / "models" / "onnx")
    return p.parse_args()


def describe(path: Path) -> None:
    """打印 ONNX 的元信息：opset、IR 版本、输入输出(含动态轴)、节点数。"""
    import onnx

    m = onnx.load(str(path))

    def dims(value_info) -> list:
        return [d.dim_param if d.HasField("dim_param") else d.dim_value
                for d in value_info.type.tensor_type.shape.dim]

    print(f"  文件     : {path}")
    print(f"  大小     : {path.stat().st_size / 1e6:.2f} MB")
    print(f"  opset    : {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    print(f"  IR 版本  : {m.ir_version}   producer: {m.producer_name} {m.producer_version}")
    for i in m.graph.input:
        print(f"  输入     : {i.name}  {dims(i)}")
    for o in m.graph.output:
        print(f"  输出     : {o.name}  {dims(o)}")
    print(f"  节点数   : {len(m.graph.node)}")


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"[x] 找不到权重: {weights}")

    print(f"[1/3] 加载 {weights.name}")
    model = YOLO(str(weights))

    print(f"[2/3] 导出 ONNX  imgsz={args.imgsz}  opset={args.opset}  "
          f"dynamic={args.dynamic}  simplify={not args.no_simplify}")
    t0 = time.perf_counter()
    raw = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        device="cpu",
    )
    cost = time.perf_counter() - t0
    raw_path = Path(raw).resolve()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = "dyn" if args.dynamic else f"b{args.batch}"
    target = args.outdir / f"{weights.stem}_{args.imgsz}_{tag}.onnx"
    if target.exists():
        target.unlink()
    shutil.move(str(raw_path), str(target))

    print(f"[3/3] 导出完成，耗时 {cost:.1f}s")
    describe(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
