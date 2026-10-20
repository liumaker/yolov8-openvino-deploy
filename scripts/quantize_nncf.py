"""NNCF post-training quantization (INT8) for the YOLOv8 OpenVINO IR.

  python scripts/quantize_nncf.py                       # default: 300 calib imgs, performance preset
  python scripts/quantize_nncf.py --preset mixed        # accuracy-oriented weights
  python scripts/quantize_nncf.py --subset 100          # quick smoke test

Calibration images come from data/coco/manifest.json ("calib": coco128 + held-out
val2017 images), so none of them are in the mAP evaluation set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 项目根目录：当前脚本位于 scripts/ 下，向上退一级即项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 把根目录加入模块搜索路径，便于 import src.*
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # Ultralytics 配置目录，避免写进用户家目录
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")  # 关闭 OpenVINO 遥测上报

import nncf  # noqa: E402  # NNCF：OpenVINO 的神经网络压缩框架，负责量化
import openvino as ov  # noqa: E402  # OpenVINO：读取/保存 IR 模型

from src.preprocess import letterbox, read_image  # noqa: E402  # 图像读取与 letterbox 前处理


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", type=Path, default=ROOT / "models" / "ir" / "yolov8n_640_b1_fp32.xml",
                    help="输入的 FP32 IR 模型 (xml)，默认取 models/ir 下的 yolov8n_640_b1_fp32.xml")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ir" / "yolov8n_640_b1_int8.xml",
                    help="输出的 INT8 量化模型路径，默认写到 models/ir/yolov8n_640_b1_int8.xml")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "coco" / "manifest.json",
                    help="COCO 数据集清单，其中 calib 字段提供标定图像")
    ap.add_argument("--subset", type=int, default=300,
                    help="用于标定的图像数量（不足标定集总数时均匀抽样），0 表示用满")
    ap.add_argument("--preset", choices=["performance", "mixed"], default="performance",
                    help="量化预设：performance=默认偏速度，mixed=权重用更细粒度保精度")
    ap.add_argument("--mode", choices=["", "sym", "asym"], default="",
                    help="量化模式：sym=对称 INT8，asym=非对称 INT8，留空由 NNCF 自动决定")
    ap.add_argument("--ignore", default="",
                    help="跳过量化的算子名字正则（保持 FP32），例如 model\\.22/")
    ap.add_argument("--batch", type=int, default=1, help="校准时的批大小（>1 会把单图重复堆叠成一批）")
    ap.add_argument("--max-drop", type=float, default=0.0,
                    help=">0 则启用精度感知量化，该值为允许的 mAP 下降比例")
    ap.add_argument("--val-images", type=int, default=150,
                    help="精度感知量化用于评估的图像数量")
    ap.add_argument("--val-source", choices=["eval", "calib"], default="eval",
                    help="精度感知量化的图像来源池：eval=从 eval-500 抽，calib=只用 calib 里的 val2017（使 eval 集完全不可见）")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    calib = [Path(p) for p in manifest["calib"]]
    # 用 --subset 均匀抽样，限制标定图像数量；标定集够多时无需用满
    if args.subset and args.subset < len(calib):
        step = len(calib) / args.subset
        calib = [calib[int(i * step)] for i in range(args.subset)]
    print(f"[nncf] calibration images: {len(calib)}")
    print(f"[nncf] first: {calib[0].name}  last: {calib[-1].name}")

    # 承载 OpenVINO 模型的运行时对象，用于读取 IR
    core = ov.Core()
    model = core.read_model(str(args.ir))  # 读入 FP32 IR 模型（含权重只有 xml+bin 里的 bin）
    # 从模型输入张量形状推断输入边长（如 640），以便对图像做 letterbox 缩放
    in_size = int(model.inputs[0].get_partial_shape()[-1].get_length())

    # NNCF 校准数据集的自定义变换函数：每个样本读图 + letterbox 到模型输入尺寸
    def transform(path: Path):
        x, _ = letterbox(read_image(path), new_size=in_size)
        if args.batch > 1:  # 需要 batch > 1 时，把同一张图重复堆叠成一批
            x = np.repeat(x, args.batch, axis=0)
        return x

    # NNCF 标定数据集：把图像列表 + 变换函数封装成校准集
    dataset = nncf.Dataset(calib, transform)
    # preset：PERFORMANCE（默认，偏速度）或 MIXED（对权重用更精细的量化以保精度）
    preset = nncf.QuantizationPreset.PERFORMANCE if args.preset == "performance" else nncf.QuantizationPreset.MIXED
    # 量化模式：sym = 对称 INT8，asym = 非对称 INT8；默认 None 由 NNCF 根据 preset 自动决定
    mode = None
    if args.mode == "sym":
        mode = nncf.QuantizationMode.INT8_SYM
    elif args.mode == "asym":
        mode = nncf.QuantizationMode.INT8_ASYM

    # --ignore 指定要“跳过量化、保持 FP32”的层：先按正则匹配 Convolution / MatMul 名字
    ignored = None
    if args.ignore:
        import re

        pat = re.compile(args.ignore)
        names = [o.get_friendly_name() for o in model.get_ops()
                 if o.get_type_name() in ("Convolution", "MatMul") and pat.search(o.get_friendly_name())]
        print(f"[nncf] ignoring {len(names)} ops matching {args.ignore!r}: {names[:6]} ...")
        ignored = nncf.IgnoredScope(names=names)

    t0 = time.perf_counter()
    if args.max_drop > 0:
        # ---------- 精度感知量化：在允许的 mAP 下降范围内自动选择哪些层量化 ----------
        from src.coco_eval import CocoRef, evaluate_predictor

        ref = CocoRef()
        if args.val_source == "calib":
            # 只有 val2017 图像在 instances_val2017.json 里有 ground truth，可从 calib 池里选
            pool = [Path(p) for p in manifest["calib"] if Path(p).parent.name == "val2017"]
            val_items = [{"id": int(p.stem), "file_name": p.name} for p in sorted(pool)]
        else:
            val_items = ref.items(args.val_images)  # 从 eval-500 里抽一部分做验证
        print(f"[nncf] accuracy-aware mode: max_drop={args.max_drop} "
              f"on {len(val_items)} held-out images ({args.val_source})")

        # NNCF 的验证回调：编译模型、跑 COCO 评估、打印 AP，并把 AP 作为精度指标返回
        def validation_fn(qmodel, data=None):
            # NNCF 传入的通常是已编译好的模型，但也会传入普通 ov.Model，需兼容两种情况
            try:
                compiled = ov.Core().compile_model(qmodel, "CPU")
            except (TypeError, RuntimeError):
                compiled = qmodel
            # 第二个参数是待评估样本迭代器：全部样本，或按层逐个评估时的单样本切片
            try:
                items = list(data) if data is not None else val_items
            except TypeError:
                items = val_items
            if not items:
                items = val_items
            stats = evaluate_predictor(lambda x: compiled(x)[0], in_size, ref, items,
                                       conf=0.02, iou=0.7)
            print(f"[nncf][val] n={len(items)} AP={stats['AP']:.4f} AP50={stats['AP50']:.4f}", flush=True)
            return float(stats["AP"]), None

        val_ds = nncf.Dataset(
            val_items,
            lambda it: letterbox(read_image(ref.image_path(it)), new_size=in_size)[0],
        )
        # 精度感知量化：给定允许的绝对 AP 下降量（ABSOLUTE），NNCF 逐步尝试并保留未超标的量化方案
        q_model = nncf.quantize_with_accuracy_control(
            model, dataset, val_ds, validation_fn,
            max_drop=args.max_drop, drop_type=nncf.DropType.ABSOLUTE,
            preset=preset, ignored_scope=ignored, subset_size=len(calib),
        )
    else:
        # ---------- 普通后训练量化：直接按标定集统计，一次到位 ----------
        q_model = nncf.quantize(model, dataset, preset=preset, mode=mode,
                                ignored_scope=ignored, subset_size=len(calib))
    dt = time.perf_counter() - t0

    # 保存量化后的 INT8 模型：只存 xml（结构）+ bin（权重），不额外做 FP16 压缩
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(q_model, str(args.out), compress_to_fp16=False)
    bin_path = args.out.with_suffix(".bin")
    print(f"[nncf] quantized in {dt:.1f}s -> {args.out} "
          f"({args.out.stat().st_size / 1024:.0f} KB) + bin {bin_path.stat().st_size / 1e6:.2f} MB")

    # 统计量化后各类算子数量，用来观察到底有多少层被真正量化成 INT8
    from collections import Counter

    ops = Counter()
    for op in q_model.get_ops():
        ops[op.get_type_name()] += 1
    print("[nncf] top ops after quantization:")
    for name, n in ops.most_common(12):
        print(f"  {name:<14} {n}")

    # 把量化元信息（来源、输出、标定量、preset、耗时、体积、算子统计）写入同名 .ir.json
    meta = {
        "source_ir": str(args.ir),
        "output_ir": str(args.out),
        "calib_images": len(calib),
        "preset": args.preset,
        "seconds": round(dt, 1),
        "xml_kb": round(args.out.stat().st_size / 1024, 1),
        "bin_mb": round(bin_path.stat().st_size / 1e6, 3),
        "ops": dict(ops),
    }
    args.out.with_suffix(".ir.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
