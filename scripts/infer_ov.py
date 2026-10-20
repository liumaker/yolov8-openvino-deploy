#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 3 步：OpenVINO IR 端到端推理（自己写 letterbox + 解码 + NMS）。

流程
    读图 -> letterbox(640) -> IR 推理 -> 解码 -> 按类 NMS -> 映射回原图 -> 画图

验收
    与 ultralytics 跑同一份静态 ONNX 的结果逐框对比（类别、坐标、分数）。

用法
    python scripts/infer_ov.py
    python scripts/infer_ov.py --ir models/ir/yolov8n_640_b1_fp16.xml --image data/samples/bus.jpg
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ov_engine import OVEngine          # noqa: E402
from src.postprocess import decode          # noqa: E402
from src.preprocess import read_image, write_image, letterbox   # noqa: E402

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVINO 端到端推理")
    p.add_argument("--ir", type=Path, default=ROOT / "models" / "ir" / "yolov8n_640_b1_fp32.xml")
    p.add_argument("--onnx", type=Path, default=ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx",
                   help="仅用于读类别名、以及与 ultralytics 做对照")
    p.add_argument("--image", type=Path, default=ROOT / "data" / "samples" / "bus.jpg")
    p.add_argument("--device", default="CPU")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--bench", type=int, default=20, help="计时迭代次数")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "bus_ov.jpg")
    p.add_argument("--no-ref", action="store_true", help="跳过 ultralytics 对照")
    return p.parse_args()


def load_names(onnx_path: Path) -> dict[int, str]:
    """类别名藏在 ONNX 的 metadata_props['names'] 里，形如 "{0: 'person', ...}"。"""
    try:
        import onnx
        meta = onnx.load(str(onnx_path))
        for prop in meta.metadata_props:
            if prop.key == "names":
                raw = ast.literal_eval(prop.value)
                return {int(k): str(v) for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 读类别名失败: {exc}")
    return {}


def draw(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
         classes: np.ndarray, names: dict[int, str]) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        color = PALETTE[int(c) % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{names.get(int(c), int(c))} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(out, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def compare_with_ultralytics(args, boxes, scores, classes) -> dict:
    """同一份静态 ONNX 让 ultralytics 跑一遍，按类别 + IoU 匹配后比坐标和分数。"""
    from ultralytics import YOLO

    model = YOLO(str(args.onnx), task="detect")
    r = model.predict(str(args.image), imgsz=640, conf=args.conf, iou=args.iou, verbose=False)[0]
    ref_b = r.boxes.xyxy.numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
    ref_s = r.boxes.conf.numpy() if len(r.boxes) else np.zeros((0,), np.float32)
    ref_c = r.boxes.cls.numpy().astype(int) if len(r.boxes) else np.zeros((0,), np.int64)

    iou = iou_matrix(boxes, ref_b)
    matched, coord_d, conf_d = [], [], []
    for i, c in enumerate(classes):
        cand = np.flatnonzero(ref_c == c)
        if cand.size == 0:
            continue
        j = cand[np.argmax(iou[i, cand])]
        if iou[i, j] >= 0.5:
            matched.append((i, j))
            coord_d.append(np.abs(boxes[i] - ref_b[j]).max())
            conf_d.append(abs(scores[i] - ref_s[j]))

    return {
        "n_ours": int(len(boxes)),
        "n_ref": int(len(ref_b)),
        "matched": len(matched),
        "max_coord_diff": float(max(coord_d)) if coord_d else None,
        "max_conf_diff": float(max(conf_d)) if conf_d else None,
    }


def main() -> int:
    args = parse_args()
    names = load_names(args.onnx)

    img = read_image(args.image)
    print(f"图片 {args.image.name} {img.shape[1]}x{img.shape[0]}")

    engine = OVEngine(args.ir, device=args.device)
    engine.warmup()
    print(f"引擎 {engine}  输入 {engine.input.get_element_type()}")

    x, meta = letterbox(img, 640)
    print(f"letterbox: ratio={meta.ratio:.4f} pad=({meta.pad_w},{meta.pad_h}) 张量 {x.shape}")

    # ---- 单次推理 ----
    t0 = time.perf_counter()
    pred = engine.infer(x)
    t_infer = (time.perf_counter() - t0) * 1e3
    boxes_net, scores, classes = decode(pred, args.conf, args.iou)
    boxes = meta.to_original(boxes_net)

    print(f"\n推理 {t_infer:.1f} ms  输出 {pred.shape} -> 检出 {len(boxes)} 个框")
    print(f"{'#':>2}  {'class':<12} {'conf':>5}  {'x1':>6} {'y1':>6} {'x2':>6} {'y2':>6}")
    for i, ((x1, y1, x2, y2), s, c) in enumerate(zip(boxes, scores, classes)):
        print(f"{i:>2}  {names.get(int(c), str(int(c))):<12} {s:>5.3f}  "
              f"{x1:>6.1f} {y1:>6.1f} {x2:>6.1f} {y2:>6.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_image(args.out, draw(img, boxes, scores, classes, names))
    print(f"可视化写入 {args.out}")

    # ---- 耗时分解 ----
    pre_ms, inf_ms, post_ms = [], [], []
    for _ in range(args.bench):
        t = time.perf_counter(); x2, _ = letterbox(img, 640); pre_ms.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); p2 = engine.infer(x2); inf_ms.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); decode(p2, args.conf, args.iou); post_ms.append((time.perf_counter() - t) * 1e3)

    def stat(v):
        return (statistics.mean(v), statistics.median(v), sorted(v)[int(len(v) * 0.95) - 1])

    e2e = statistics.mean(pre_ms) + statistics.mean(inf_ms) + statistics.mean(post_ms)
    print(f"\n耗时分解（{args.bench} 次，CPU 单线程调用）")
    print(f"  pre  {stat(pre_ms)[0]:6.2f} ms (p50 {stat(pre_ms)[1]:.2f} / p95 {stat(pre_ms)[2]:.2f})")
    print(f"  infer{stat(inf_ms)[0]:6.2f} ms (p50 {stat(inf_ms)[1]:.2f} / p95 {stat(inf_ms)[2]:.2f})")
    print(f"  post {stat(post_ms)[0]:6.2f} ms (p50 {stat(post_ms)[1]:.2f} / p95 {stat(post_ms)[2]:.2f})")
    print(f"  e2e  {e2e:6.2f} ms  -> {1000 / e2e:.1f} FPS")

    # ---- 对照 ----
    cmp = None
    if not args.no_ref:
        print("\n[对照] ultralytics 跑同一份静态 ONNX")
        cmp = compare_with_ultralytics(args, boxes, scores, classes)
        print(f"  我们 {cmp['n_ours']} 框 / 参考 {cmp['n_ref']} 框，匹配 {cmp['matched']} 个")
        if cmp["max_coord_diff"] is not None:
            print(f"  max|Δ坐标| = {cmp['max_coord_diff']:.4f} px   max|Δconf| = {cmp['max_conf_diff']:.2e}")

    report = {
        "ir": str(args.ir), "image": str(args.image), "device": args.device,
        "conf": args.conf, "iou": args.iou,
        "ratio": meta.ratio, "pad": [meta.pad_w, meta.pad_h],
        "detections": [
            {"class": int(c), "name": names.get(int(c), ""), "conf": float(s),
             "box": [float(v) for v in b]}
            for b, s, c in zip(boxes, scores, classes)
        ],
        "timing_ms": {
            "pre_mean": statistics.mean(pre_ms), "infer_mean": statistics.mean(inf_ms),
            "post_mean": statistics.mean(post_ms), "e2e_mean": e2e,
            "infer_p95": stat(inf_ms)[2],
        },
        "compare": cmp,
    }
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告写入 {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
