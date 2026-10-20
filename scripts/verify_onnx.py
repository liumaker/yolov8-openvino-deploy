#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 1 步验收：ONNX 图检查 + ONNXRuntime 形状验证 + 与 PyTorch 逐元素数值对齐。

为什么这样测
- 端到端比对（.pt vs .onnx 各跑一次 predict）会被 ultralytics 的预处理差异污染：
  .pt 与动态 ONNX 用 LetterBox(auto=True) 做最小矩形填充，静态 ONNX 必须 auto=False
  补成正方形，同一张图喂进网络的像素其实不一样，框自然对不上；
- 所以主判据是「同一个 float32 输入张量，PyTorch 与 ONNX 的输出差多少」，
  这才是模型本身是否等价；端到端那一层放到方形图上做，排除 letterbox 干扰。

用法
    python scripts/verify_onnx.py --onnx models/onnx/yolov8n_640_b1.onnx
    python scripts/verify_onnx.py --onnx models/onnx/yolov8n_640_dyn.onnx --probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校验导出的 ONNX 模型")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8n.pt")
    p.add_argument("--image", type=Path, default=ROOT / "data" / "samples" / "bus.jpg")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--tol", type=float, default=2e-2, help="原始输出最大绝对误差阈值")
    p.add_argument("--probe", action="store_true", help="非方形/多 batch 输入验证动态轴")
    p.add_argument("--json", type=Path, default=None, help="元信息落盘路径")
    return p.parse_args()


def dim_label(d):
    return d.dim_param if d.HasField("dim_param") else d.dim_value


def input_dims(m: onnx.ModelProto) -> list:
    return [dim_label(d) for d in m.graph.input[0].type.tensor_type.shape.dim]


def expected_anchors(h: int, w: int) -> int:
    """YOLOv8 三个输出层 stride 8/16/32，网格点数求和。"""
    total, stride = 0, 8
    for _ in range(3):
        total += (h // stride) * (w // stride)
        stride *= 2
    return total


def check_graph(path: Path) -> dict:
    m = onnx.load(str(path))
    onnx.checker.check_model(m)  # 不合法会抛异常
    inferred = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)

    ops = Counter(n.op_type for n in m.graph.node)
    opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    in_dims = input_dims(m)
    out_dims = [dim_label(d) for d in m.graph.output[0].type.tensor_type.shape.dim]
    dynamic_axes = {f"input[{i}]": d for i, d in enumerate(in_dims) if isinstance(d, str) and d}

    print("[1/4] 图检查")
    print("      checker         : OK")
    print(f"      opset / IR      : {opset} / {m.ir_version}")
    print(f"      producer        : {m.producer_name} {m.producer_version}")
    print(f"      输入            : {m.graph.input[0].name} {in_dims}")
    print(f"      输出            : {m.graph.output[0].name} {out_dims}")
    print(f"      动态轴          : {dynamic_axes or '无（全静态）'}")
    print(f"      节点数          : {len(m.graph.node)}  中间张量 {len(inferred.graph.value_info)}")
    print(f"      算子 Top10      : {ops.most_common(10)}")

    return {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "opset": opset,
        "ir_version": m.ir_version,
        "input_name": m.graph.input[0].name,
        "input_dims": [str(d) for d in in_dims],
        "output_name": m.graph.output[0].name,
        "output_dims": [str(d) for d in out_dims],
        "dynamic_axes": dynamic_axes,
        "num_nodes": len(m.graph.node),
        "op_counts": dict(ops),
    }


def check_runtime(path: Path, imgsz: int, nc: int, probe: bool) -> dict:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    o_name = sess.get_outputs()[0].name

    print("\n[2/4] ONNXRuntime 形状验证")
    x = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    y = sess.run([o_name], {i_name: x})[0]
    want = (1, 4 + nc, expected_anchors(imgsz, imgsz))
    assert tuple(y.shape) == want, f"输出形状 {y.shape} != 期望 {want}"
    assert np.isfinite(y).all(), "输出含 NaN/Inf"
    print(f"      输入 {x.shape} -> 输出 {y.shape}（期望 {want}）")

    result = {"runtime": "onnxruntime " + ort.__version__, "output_shape": list(y.shape)}

    if probe:
        print("\n[3/4] 动态轴探针")
        probes = []
        for shape in [(1, 3, 480, 640), (2, 3, 640, 640), (1, 3, 640, 640)]:
            try:
                xx = np.random.randn(*shape).astype(np.float32)
                yy = sess.run([o_name], {i_name: xx})[0]
                want_p = (shape[0], 4 + nc, expected_anchors(shape[2], shape[3]))
                ok = tuple(yy.shape) == want_p
                print(f"      {shape} -> {tuple(yy.shape)}  {'OK' if ok else 'MISMATCH ' + str(want_p)}")
                probes.append({"input": list(shape), "output": list(yy.shape), "ok": ok})
            except Exception as exc:  # 静态模型在这里会报维度不匹配
                print(f"      {shape} -> 失败: {type(exc).__name__}: {str(exc)[:120]}")
                probes.append({"input": list(shape), "error": str(exc)[:200], "ok": False})
        result["dynamic_probes"] = probes

    return result


def build_input(image: Path, imgsz: int) -> np.ndarray:
    """复刻 ultralytics 预处理：BGR->RGB、letterbox 到正方形、HWC->CHW、/255。"""
    import cv2
    from ultralytics.data.augment import LetterBox

    im = cv2.imread(str(image))
    if im is None:
        raise SystemExit(f"[x] 读不到图片: {image}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = LetterBox(imgsz, auto=False, stride=32)(image=im)
    return np.ascontiguousarray(im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def torch_raw(weights: Path, x: np.ndarray) -> np.ndarray:
    """让 PyTorch 走与导出时相同的 export 分支，拿到 (1, 4+nc, anchors) 原始输出。"""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights), verbose=False)
    model.model.fuse()
    model.model.model[-1].export = True  # Detect head 输出拼接后的张量
    model.model.eval()
    with torch.no_grad():
        y = model.model(torch.from_numpy(x))
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y.numpy()


def check_raw_equivalence(path: Path, weights: Path, image: Path,
                          imgsz: int, tol: float) -> dict:
    print("\n[3/4] 原始输出逐元素对齐（同一输入张量，不经过两次预处理）")
    x = build_input(image, imgsz)
    y_pt = torch_raw(weights, x)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    assert y_pt.shape == y_onnx.shape, f"形状不一致 {y_pt.shape} vs {y_onnx.shape}"
    diff = np.abs(y_pt - y_onnx)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    # 框坐标在 0~640 像素量级，通道 4+ 是类别 logits
    box_d = float(diff[:, :4].max())
    cls_d = float(diff[:, 4:].max())
    ok = max_d <= tol

    print(f"      输入            : {x.shape}  范围 [{x.min():.2f}, {x.max():.2f}]")
    print(f"      输出            : {y_pt.shape} == {y_onnx.shape}")
    print(f"      最大绝对误差    : {max_d:.6f}   平均 {mean_d:.8f}")
    print(f"      其中 框坐标     : {box_d:.6f}   类别 logits {cls_d:.6f}")
    print(f"      结论            : {'PASS' if ok else 'FAIL'}（阈值 {tol}）")

    return {"max_abs_diff": max_d, "mean_abs_diff": mean_d,
            "box_max_diff": box_d, "cls_max_diff": cls_d,
            "tolerance": tol, "verdict": "PASS" if ok else "FAIL"}


def check_e2e(path: Path, weights: Path, image: Path, imgsz: int,
              conf: float, iou: float) -> dict:
    """端到端比对：先把图补成正方形，让 .pt 与 .onnx 的 letterbox 策略一致。"""
    import cv2
    from ultralytics import YOLO

    im = cv2.imread(str(image))
    h, w = im.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 114, np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = im

    print("\n[4/4] 端到端比对（正方形输入，conf=%.2f iou=%.2f）" % (conf, iou))
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
    r_pt = YOLO(str(weights), verbose=False).predict(canvas, **kw)[0]
    r_onnx = YOLO(str(path), verbose=False).predict(canvas, **kw)[0]

    b_pt, b_onnx = r_pt.boxes, r_onnx.boxes
    n_pt, n_onnx = len(b_pt), len(b_onnx)
    out = {"num_boxes_pt": n_pt, "num_boxes_onnx": n_onnx}
    print(f"      PyTorch {n_pt} 个框 / ONNX {n_onnx} 个框")

    if n_pt and n_onnx:
        n = min(n_pt, n_onnx)
        d_xyxy = float(np.abs(b_pt.xyxy[:n].numpy() - b_onnx.xyxy[:n].numpy()).max())
        d_conf = float(np.abs(b_pt.conf[:n].numpy() - b_onnx.conf[:n].numpy()).max())
        same_cls = bool((b_pt.cls[:n].numpy() == b_onnx.cls[:n].numpy()).all())
        print(f"      框坐标最大偏差  : {d_xyxy:.4f} px")
        print(f"      置信度最大偏差  : {d_conf:.6f}")
        print(f"      类别一致        : {same_cls}")
        good = n_pt == n_onnx and d_xyxy < 1.0 and d_conf < 1e-3 and same_cls
        print(f"      结论            : {'PASS' if good else 'DIFF'}")
        out.update({"max_xyxy_diff_px": round(d_xyxy, 4), "max_conf_diff": d_conf,
                    "classes_match": same_cls, "verdict": "PASS" if good else "DIFF"})
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    path = args.onnx.resolve()
    if not path.is_file():
        raise SystemExit(f"[x] 找不到 ONNX: {path}")

    nc = 80
    if args.weights.is_file():
        from ultralytics import YOLO
        nc = len(YOLO(str(args.weights), verbose=False).names)

    meta = check_graph(path)
    meta.update(check_runtime(path, args.imgsz, nc, args.probe))

    verdicts = []
    if args.weights.is_file() and args.image.is_file():
        meta["raw_equivalence"] = check_raw_equivalence(path, args.weights, args.image,
                                                        args.imgsz, args.tol)
        verdicts.append(meta["raw_equivalence"]["verdict"])
        meta["e2e_equivalence"] = check_e2e(path, args.weights, args.image, args.imgsz,
                                            args.conf, args.iou)
        verdicts.append(meta["e2e_equivalence"]["verdict"])
    else:
        print("\n[3/4][4/4] 跳过对齐：缺少权重或测试图")

    out_json = args.json or path.with_suffix(".meta.json")
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n元信息写入 {out_json}")
    print("总体结论：" + ("PASS" if verdicts and all(v == "PASS" for v in verdicts) else "FAIL"))
    return 0 if verdicts and all(v == "PASS" for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
