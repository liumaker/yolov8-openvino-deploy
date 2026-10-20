"""Compute COCO bbox mAP for an OpenVINO IR or ONNX model on the prepared val2017 subset.

Usage:
  python scripts/eval_map.py --model models/ir/yolov8n_640_b1_fp32.xml
  python scripts/eval_map.py --model models/ir/yolov8n_640_b1_int8.xml --out results/map_int8.json

Pipeline is the deployed one (own letterbox + own decode/NMS), not ultralytics' predictor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

from src.coco_eval import CocoRef, coco_stats, predict_items  # noqa: E402


def build_predictor(model_path: Path, device: str, num_streams: int | None):
    """-> (infer_fn(np.ndarray[1,3,S,S]) -> raw output, input_size S)."""
    if model_path.suffix.lower() == ".xml":
        from src.ov_engine import OVEngine

        eng = OVEngine(model_path, device=device, num_streams=num_streams)
        eng.warmup(3)
        return eng.infer, eng.input_shape[-1]
    if model_path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return (lambda x: sess.run(None, {name: x})[0]), int(sess.get_inputs()[0].shape[-1])
    raise ValueError(f"unsupported model: {model_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--num-streams", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="COCO protocol keeps low-confidence boxes")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU, COCO protocol")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="only first N eval images (debug)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ref = CocoRef()
    items = ref.items(args.limit)
    infer, size = build_predictor(args.model, args.device, args.num_streams)
    print(f"[eval] model={args.model.name} device={args.device} input={size} images={len(items)}")

    t0 = time.perf_counter()
    results = predict_items(infer, size, items, ref, args.conf, args.iou, args.max_det)
    dt = time.perf_counter() - t0
    print(f"[eval] {len(results)} detections in {dt:.2f}s ({len(items) / dt:.1f} img/s)")

    if not results:
        print("[eval] no detections at all")
        return 1
    stats = coco_stats(ref, results, items)
    print("\n[eval] summary: " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    report = {
        "model": str(args.model),
        "device": args.device,
        "images": len(items),
        "detections": len(results),
        "conf": args.conf,
        "nms_iou": args.iou,
        "seconds": round(dt, 3),
        "ms_per_img": round(dt / len(items) * 1000, 2),
        "metrics": stats,
    }
    out = args.out or (ROOT / "results" / f"map_{args.model.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
