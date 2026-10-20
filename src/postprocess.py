#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOv8 后处理：解码 + 手写 NMS。

为什么要自己写：cv2.dnn.NMSBoxes 一行就能出结果，但面试问的"NMS 怎么实现的"
就答不上来；而且它不支持按类别分开抑制，混合类别时会互相压制。

输出张量 (1, 84, 8400) 的含义
    84  = 4 个框参数 (cx, cy, w, h) + 80 个类别分数（已过 sigmoid）
    8400 = 80x80 + 40x40 + 20x20 三个尺度的网格点数
    坐标已经是**输入图像素**（导出时乘过 stride），不要再乘 8/16/32
    YOLOv8 是 anchor-free 且没有 objectness，置信度 = max(类别分数)
"""
from __future__ import annotations

import numpy as np


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    """(cx,cy,w,h) -> (x1,y1,x2,y2)，支持 (N,4)。"""
    out = np.empty_like(b, dtype=np.float32)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """单类别贪心 NMS，返回保留框的下标（按分数降序）。

    流程：分数排序 -> 取最高分 -> 算它与剩下所有框的 IoU -> 删掉 IoU 超阈值的
    -> 重复，直到没有剩余框。复杂度 O(n^2)，但候选框已经被置信度阈值砍到几十个。
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]

    return np.asarray(keep, dtype=np.int64)


def decode(pred: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45,
           max_det: int = 300):
    """原始输出 -> (boxes_xyxy, scores, classes)，坐标仍是网络输入尺度。"""
    p = np.asarray(pred, dtype=np.float32)
    if p.ndim == 3:          # (1, 84, 8400)
        p = p[0]
    p = p.T                  # (8400, 84)

    boxes_xywh = p[:, :4]
    cls_scores = p[:, 4:]
    classes = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), classes]

    mask = scores > conf_thres
    if not mask.any():
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
                np.empty((0,), np.int64))

    boxes = xywh2xyxy(boxes_xywh[mask])
    scores = scores[mask]
    classes = classes[mask]

    # 按类别分别 NMS：不同类的框不该互相压制
    kept: list[np.ndarray] = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        kept.append(idx[nms(boxes[idx], scores[idx], iou_thres)])
    keep = np.concatenate(kept) if kept else np.empty((0,), np.int64)

    keep = keep[np.argsort(scores[keep])[::-1]][:max_det]
    return boxes[keep], scores[keep], classes[keep]
