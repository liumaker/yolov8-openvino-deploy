#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenVINO Runtime 的最小封装：读 IR、编译、预热、同步推理。

后面做异步/多路视频流时，会在这个类上再加 infer_async / start_async，
现在先把同步路径跑通。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class OVEngine:
    """OpenVINO 推理引擎封装。

    职责：把「读 IR 模型 → 编译到指定设备 → 预热 → 同步推理」整条链路
    封装起来，上层代码只需要拿到 OVEngine 实例，调 infer() 即可，无需
    直接接触 openvino 的 Core / CompiledModel 等 API。
    """

    def __init__(self, xml: Path | str, device: str = "CPU", num_streams: int | None = None):
        """初始化：读取并编译模型，记录输入/输出 shape 和编译耗时。

        Args:
            xml:         IR 模型文件路径（.xml，同目录下需有对应 .bin 权重）
            device:     推理设备，默认 "CPU"，也可传 "GPU"、"AUTO" 等
            num_streams:并行推理流数量（吞吐优化用）；None 表示用设备默认值
        """
        # openvino 包较重，延迟 import：只在真正创建 OVEngine 时才引入
        import openvino as ov

        self.xml = Path(xml)                      # 保存模型路径，便于调试/展示

        # Core 是 OpenVINO 的入口对象，负责读取与编译模型（一个 Core 可复用给多个模型）
        core = ov.Core()
        self.model = core.read_model(str(self.xml))  # 读 IR：.xml 结构 + 同目录 .bin 权重

        # 编译配置：设置推理流数量，让编译器和运行时按此调度执行
        config = {}
        if num_streams is not None:
            config["NUM_STREAMS"] = num_streams

        # 记录编译耗时（compile_model 会做模型转换/内核选择，可能较慢）
        t0 = time.perf_counter()
        self.compiled = core.compile_model(self.model, device, config)
        self.compile_ms = (time.perf_counter() - t0) * 1e3  # 秒转毫秒

        # 取第 0 个输入/输出端口（本仓库是单输入单输出模型，够用）
        self.input = self.compiled.input(0)
        self.output = self.compiled.output(0)

        # PartialShape 里的维度是 Dimension 对象：is_static 判定它是否是动态维。
        # 静态维取实际长度；动态维（如 batch 可变）记为 -1，方便上层识别。
        self.input_shape = tuple(
            int(d.get_length()) if d.is_static else -1 for d in self.input.get_partial_shape()
        )
        self.output_shape = tuple(
            int(d.get_length()) if d.is_static else -1 for d in self.output.get_partial_shape()
        )

    def infer(self, x: np.ndarray) -> np.ndarray:
        """同步推理。

        compiled 对象是可调用的，传入 list 即按输入端口顺序喂数据；
        用 self.output 作为 key 从返回字典中取出对应输出张量，转成 ndarray。
        """
        result = self.compiled([x])                 # 喂入输入，得到 {port: tensor} 的映射
        return np.asarray(result[self.output])      # 取出目标输出，包装成 ndarray 返回

    def warmup(self, iters: int = 3) -> None:
        """预热：跑几次空推理，让运行时完成第一次推理的初始化。

        第一次推理会触发内核选择和内存分配，若不预热就计时会偏慢；
        因此正式 Benchmark / 计时之前必须先调用本方法。
        """
        # 构造 dummy 输入：动态维（-1）按 640 填充（YOLO 默认输入尺寸），其余按真实 shape
        shape = tuple(640 if d < 0 else d for d in self.input_shape)
        dummy = np.zeros(shape, dtype=np.float32)   # 全零占位张量，只走流程不关心数值
        for _ in range(iters):
            self.infer(dummy)                       # 连续推理多次，确保内核/内存均就绪

    def __repr__(self) -> str:
        """便于调试的字符串表示：模型名、输入/输出 shape、编译耗时。"""
        return (f"OVEngine({self.xml.name}, in={self.input_shape}, out={self.output_shape}, "
                f"compile={self.compile_ms:.0f}ms)")