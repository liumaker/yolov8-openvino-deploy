#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""letterbox：保持长宽比缩放 + 补边到正方形，并记录反算所需的 ratio / pad。

不用 ultralytics 的任何代码，纯 cv2 + numpy。后处理拿到网络坐标后，
调用 LetterboxMeta.to_original() 就能映射回原图。

注意：Windows 下 cv2.imread / imwrite 走的是窄字符 API，路径里带中文（本工作区
"D:\\简历\\..."）会直接返回 None。所以统一用 imdecode/imencode + np.fromfile/tofile。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path | str) -> np.ndarray:
    """读 BGR 图，兼容中文路径。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"读不到图片: {path}")
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"不是有效图片: {path}")
    return im


def write_image(path: Path | str, im: np.ndarray) -> None:
    """写图，兼容中文路径。"""
    suffix = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, im)
    if not ok:
        raise ValueError(f"编码失败: {path}")
    buf.tofile(str(path))


@dataclass(frozen=True)
class LetterboxMeta:
    ratio: float      # 缩放比例（原图 -> 网络）
    pad_w: int        # 左侧补边宽度
    pad_h: int        # 顶部补边高度
    orig_w: int
    orig_h: int

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """把网络坐标下的 xyxy 映射回原图坐标并裁剪到图内。"""
        b = np.asarray(boxes, dtype=np.float32).copy()
        if b.size == 0:
            return b.reshape(0, 4)
        b[:, [0, 2]] -= self.pad_w
        b[:, [1, 3]] -= self.pad_h
        b /= self.ratio
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.orig_w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.orig_h)
        return b


def letterbox(im: np.ndarray, new_size: int = 640, color: int = 114):
    """BGR 图 -> (1,3,S,S) float32 张量 + 反算元信息。

    步骤：等比缩放 -> 居中贴在 S×S 灰底画布 -> BGR 转 RGB -> HWC 转 CHW -> /255。
    与 ultralytics 的 auto=False 行为一致（灰边值 114）。
    """
    h, w = im.shape[:2]
    r = min(new_size / h, new_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_h = (new_size - nh) // 2
    pad_w = (new_size - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    x = np.ascontiguousarray(x)
    return x, LetterboxMeta(ratio=r, pad_w=pad_w, pad_h=pad_h, orig_w=w, orig_h=h)
