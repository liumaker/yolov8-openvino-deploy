"""Shared COCO evaluation helpers: run a model over images the way the shipped code does,
then score the predictions with pycocotools.

Used by scripts/eval_map.py (mAP report) and scripts/quantize_nncf.py (accuracy-aware PTQ).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from src.postprocess import decode  # noqa: E402
from src.preprocess import letterbox, read_image  # noqa: E402

DATA = ROOT / "data" / "coco"


def load_class_names() -> dict[int, str]:
    import onnx

    onnx_path = sorted((ROOT / "models" / "onnx").glob("*_b1.onnx"))[0]
    m = onnx.load(str(onnx_path), load_external_data=False)
    for prop in m.metadata_props:
        if prop.key == "names":
            names = ast.literal_eval(prop.value)
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
            return dict(enumerate(names))
    raise RuntimeError("no class names in ONNX metadata")


class CocoRef:
    """Ground truth + the class-index -> category-id mapping."""

    def __init__(self, ann_path: Path | None = None):
        from pycocotools.coco import COCO

        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.coco = COCO(str(ann_path or manifest["annotations"]))
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_by_name = {c["name"]: c["id"] for c in cats}
        self.id_to_name = {c["id"]: c["name"] for c in cats}
        self.names = load_class_names()
        self.cls_to_cat = np.array([self.cat_by_name[self.names[i]] for i in range(len(self.names))])

    def items(self, limit: int = 0) -> list[dict]:
        return self.manifest["eval"][: limit or None]

    def image_path(self, item: dict) -> Path:
        return DATA / "images" / "val2017" / item["file_name"]

    def category_ids(self) -> list[int]:
        return sorted({int(c) for c in self.cls_to_cat})


def predict_items(infer, size: int, items: list[dict], ref: CocoRef,
                  conf: float = 0.001, iou: float = 0.7, max_det: int = 300) -> list[dict]:
    """Run the deployed pre/post-processing pipeline; returns COCO prediction dicts."""
    results: list[dict] = []
    for it in items:
        img = read_image(ref.image_path(it))
        x, meta = letterbox(img, new_size=size)
        boxes, scores, classes = decode(infer(x), conf, iou, max_det)
        boxes = meta.to_original(boxes)
        for b, s, c in zip(boxes, scores, classes):
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            if bw <= 0 or bh <= 0:
                continue
            results.append({
                "image_id": int(it["id"]),
                "category_id": int(ref.cls_to_cat[int(c)]),
                "bbox": [float(b[0]), float(b[1]), bw, bh],
                "score": float(s),
            })
    return results


def coco_stats(ref: CocoRef, results: list[dict], items: list[dict]) -> dict:
    from pycocotools.cocoeval import COCOeval

    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"]
    if not results:  # loadRes([]) raises IndexError; a single image can legitimately have none
        return dict.fromkeys(keys, 0.0)
    dt = ref.coco.loadRes(results)
    ev = COCOeval(ref.coco, dt, "bbox")
    ev.params.imgIds = [int(it["id"]) for it in items]
    ev.params.catIds = ref.category_ids()
    ev.evaluate()
    ev.accumulate()
    ev.summarize()  # only summarize() fills ev.stats
    return {k: float(v) for k, v in zip(keys, ev.stats[:9])}


def evaluate_predictor(infer, size: int, ref: CocoRef, items: list[dict],
                       conf: float = 0.001, iou: float = 0.7) -> dict:
    return coco_stats(ref, predict_items(infer, size, items, ref, conf, iou), items)
