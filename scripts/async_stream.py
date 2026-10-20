"""Multi-stream inference: N worker threads, each with its own InferRequest on one CompiledModel.

This is the "multiple camera streams" experiment: every stream pulls frames from its own
queue, runs inference, and decodes/NMS in the same worker (post-processing overlaps with the
other streams' inference). Reports total throughput, per-stream FPS, latency distribution and CPU load.

Why threads instead of AsyncInferQueue: in OpenVINO 2026.4.1 the Python build does not forward
`userdata` to the callback (always None), so the async-queue version cannot tell which stream a
finished request belongs to. One InferRequest per thread is the natural equivalent.

Usage:
  python scripts/async_stream.py --streams 4 --seconds 20
  python scripts/async_stream.py --streams 1                 # single-stream baseline
"""
from __future__ import annotations

import argparse  # 解析命令行参数（streams / seconds / frames / hint 等）
import json  # 将汇总结果写入 JSON 文件
import os  # 设置环境变量（关闭 OpenVINO 遥测）
import statistics  # 计算均值 / 分位数等统计数据
import sys  # 修改模块搜索路径、退出码
import threading  # 多流并发：每个流一个线程
import time  # 计时（deadline、推理耗时、系统 CPU 采样周期）
from pathlib import Path  # 路径操作

import numpy as np  # 数组与图像数据

# 项目根目录：由本文件路径上溯一级得到
ROOT = Path(__file__).resolve().parents[1]
# 把项目根目录加入模块搜索路径，才能 import src 下的代码
sys.path.insert(0, str(ROOT))
# 关闭 OpenVINO 遥测上报，避免测试期间产生额外网络/IO 开销
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import openvino as ov  # noqa: E402   OpenVINO 运行时 API
import psutil  # noqa: E402           进程/系统 CPU 占用统计

from src.postprocess import decode  # noqa: E402   YOLOv8 后处理（decode 输出 + NMS）
from src.preprocess import letterbox, read_image  # noqa: E402   读图 + 等比例填充缩放

# 测试图像的默认目录：data/coco/images/val2017
DATA = ROOT / "data" / "coco"


class CpuSampler(threading.Thread):
    """后台守护线程：周期性地采样进程 CPU 与系统 CPU 占用率。

    与流推理线程并行运行，结束时取出采样序列算均值/峰值。
    """

    def __init__(self, period: float = 0.25):
        super().__init__(daemon=True)
        self.period = period                    # 采样间隔（秒）
        self.stop_evt = threading.Event()       # 停止信号
        self.proc: list[float] = []             # 本进程 CPU 占用采样序列（%）
        self.sys: list[float] = []              # 系统整体 CPU 占用采样序列（%）
        self._p = psutil.Process()              # 当前进程的句柄

    def run(self) -> None:
        # 先调用一次 cpu_percent(None) 建立采样基准点，后续调用才返回真实增量
        self._p.cpu_percent(None)
        psutil.cpu_percent(None)
        # 循环采样：wait(period) 既做间隔定时，又能在 stop() 时立即唤醒退出
        while not self.stop_evt.wait(self.period):
            self.proc.append(self._p.cpu_percent(None))
            self.sys.append(psutil.cpu_percent(None))

    def stop(self) -> None:
        """通知线程退出并等待其收尾（最多等 2 秒）。"""
        self.stop_evt.set()
        self.join(timeout=2)


def load_frames(n: int, size: int) -> list[np.ndarray]:
    """预加载 n 张测试图片并完成前置处理。

    前置处理（letterbox + 缩放 + 转连续内存）只做一次，推理循环里直接复用，
    避免把处理耗时计入推理延迟。
    - n: 从 val2017 中取前 n 张图片
    - size: 模型输入边长（如 640），letterbox 会等比缩放并填充到该尺寸
    - 返回: 已处理好的输入数组列表，可直接作为模型输入
    """
    files = sorted((DATA / "images" / "val2017").glob("*.jpg"))[:n]
    out = []
    for f in files:
        x, _ = letterbox(read_image(f), new_size=size)
        out.append(np.ascontiguousarray(x))  # 确保内存连续，避免推理前隐式拷贝
    return out


class Stream(threading.Thread):
    """单条视频流的模拟：独占一个 InferRequest，循环推理直到截止时间。

    每个流线程做三件事：取帧 -> infer() 推理 -> decode+NMS 后处理。
    由于多线程并行，后处理可以和其它流的推理重叠执行，这正模拟了
    多路摄像头场景下 CPU 的饱和利用情况。
    """

    def __init__(self, sid: int, compiled, frames: list[np.ndarray], deadline: float,
                 conf: float, iou: float, lock: threading.Lock, sink: dict):
        super().__init__(daemon=True)
        self.sid = sid                       # 流编号（0..streams-1）
        self.compiled = compiled             # 已编译的模型，CreateInferRequest 由它创建
        self.frames = frames                 # 预处理的图片池，循环取用
        self.deadline = deadline             # 截止时间戳，到达即停止
        self.conf = conf                     # 后处理置信度阈值
        self.iou = iou                       # 后处理 NMS 的 IoU 阈值
        self.lock = lock                     # 主线程与各流共享的锁，保护 sink
        self.sink = sink                     # 共享结果收集器 {"frames":[], "lat":[], "post":[]}

    def run(self) -> None:
        # 每个线程创建自己的 InferRequest，互不共享，天然适配多流并发
        req = self.compiled.create_infer_request()
        idx = self.sid % len(self.frames)  # 各流从图片池的不同起点开始，避免完全同步
        lat: list[float] = []   # 每次推理的耗时（毫秒），含从提交到拿回结果
        post: list[float] = []  # 每次后处理（decode+NMS）的耗时（毫秒）
        while time.perf_counter() < self.deadline:
            x = self.frames[idx % len(self.frames)]
            idx += 1
            t0 = time.perf_counter()
            out = req.infer({0: x})[0]   # 单次同步推理，返回输出张量（4D）
            t1 = time.perf_counter()
            decode(out, self.conf, self.iou)   # 后处理：解码检测框 + NMS
            t2 = time.perf_counter()
            lat.append((t1 - t0) * 1000)
            post.append((t2 - t1) * 1000)
        # 结束后把本流数据合并进共享 sink（加锁防止并发写冲突）
        with self.lock:
            self.sink["frames"].append(len(lat))  # 本流处理的帧数
            self.sink["lat"].extend(lat)          # 累积所有流的推理耗时
            self.sink["post"].extend(post)        # 累积所有流的后处理耗时


def main() -> int:
    """主流程：解析参数 -> 编译模型 -> 预热 -> 起 N 条流跑测试 -> 汇总统计并落盘。"""
    # --- 命令行参数 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", type=Path, default=ROOT / "models/ir/yolov8n_640_b1_int8_aa2.xml")  # 默认用 INT8 量化模型
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--streams", type=int, default=4)   # 并发流（线程）数量
    ap.add_argument("--seconds", type=float, default=20.0)  # 测试时长（秒）
    ap.add_argument("--frames", type=int, default=64, help="pre-letterboxed images to cycle through")  # 图片池大小
    ap.add_argument("--threads", type=int, default=0, help="INFERENCE_NUM_THREADS, 0 = default")  # 推理线程数，0=自动
    ap.add_argument("--hint", default="throughput", choices=["none", "latency", "throughput"])  # 性能提示
    ap.add_argument("--conf", type=float, default=0.25)   # 置信度阈值
    ap.add_argument("--iou", type=float, default=0.45)    # NMS IoU 阈值
    ap.add_argument("--out", type=Path, default=None)     # 结果输出路径（默认按流数生成）
    args = ap.parse_args()

    # --- 读模型 + 编译 ---
    core = ov.Core()
    model = core.read_model(str(args.ir))
    # 从输入张量的偏序形状取最后一个维度 = 输入边长（如 640）
    size = int(model.inputs[0].get_partial_shape()[-1].get_length())
    cfg: dict[str, object] = {}
    if args.hint != "none":
        cfg["PERFORMANCE_HINT"] = args.hint.upper()  # 设置吞吐量/延迟性能提示
    if args.threads:
        cfg["INFERENCE_NUM_THREADS"] = args.threads  # 显式指定推理线程数
    compiled = core.compile_model(model, args.device, cfg)

    # --- 预热：先跑 3 次，触发线程池/内存分配等一次性开销，避免计入统计 ---
    frames = load_frames(args.frames, size)
    for _ in range(3):  # warmup
        compiled.create_infer_request().infer({0: frames[0]})
    print(f"[async] ir={args.ir.name} device={args.device} streams={args.streams} "
          f"frames={len(frames)} seconds={args.seconds} hint={args.hint} threads={args.threads or 'auto'}")

    # --- 启动测试：CPU 采样器 + 流线程 ---
    lock = threading.Lock()
    sink = {"frames": [], "lat": [], "post": []}  # 各流共享的结果收集器
    start = time.perf_counter()
    deadline = start + args.seconds
    sampler = CpuSampler()
    sampler.start()  # 后台开始采样 CPU 占用

    threads = [Stream(i, compiled, frames, deadline, args.conf, args.iou, lock, sink)
               for i in range(args.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # 等待所有流跑完（所有流共享同一 deadline）
    elapsed = time.perf_counter() - start  # 实际运行时长 = 截止时间 - 开始时间
    sampler.stop()

    # --- 汇总统计 ---
    total = sum(sink["frames"])             # 所有流处理的帧总数
    lat = sorted(sink["lat"])               # 排序后便于求分位数
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # 求 p 分位数（近似）: noqa: E731
    summary = {
        "ir": str(args.ir),
        "device": args.device,
        "streams": args.streams,
        "threads": args.threads or None,
        "hint": args.hint,
        "seconds": round(elapsed, 2),
        "frames_total": total,                                     # 总帧数
        "fps_total": round(total / elapsed, 2),                    # 总吞吐量（帧/秒）
        "fps_per_stream": round(total / elapsed / args.streams, 2),  # 平均每流吞吐量
        "frames_per_stream": sink["frames"],                       # 各流处理的帧数分布
        "infer_ms_mean": round(statistics.fmean(lat), 2) if lat else None,  # 平均推理延迟
        "infer_ms_p95": round(q(0.95), 2) if lat else None,        # P95 推理延迟
        "post_ms_mean": round(statistics.fmean(sink["post"]), 2) if sink["post"] else None,  # 平均后处理耗时
        "cpu_proc_pct_mean": round(statistics.fmean(sampler.proc), 1) if sampler.proc else None,  # 进程 CPU 平均占用
        "cpu_proc_pct_max": round(max(sampler.proc), 1) if sampler.proc else None,               # 进程 CPU 峰值占用
        "cpu_sys_pct_mean": round(statistics.fmean(sampler.sys), 1) if sampler.sys else None,    # 系统 CPU 平均占用
        "logical_cores": psutil.cpu_count(),                       # 逻辑核心数（用于判断是否满载）
    }
    # 控制台打印核心指标
    print("[async] " + "  ".join(
        f"{k}={summary[k]}" for k in ("frames_total", "fps_total", "fps_per_stream",
                                      "infer_ms_mean", "infer_ms_p95", "post_ms_mean",
                                      "cpu_proc_pct_mean", "cpu_sys_pct_mean")))

    # --- 结果落盘 ---
    out = args.out or (ROOT / "results" / f"async_s{args.streams}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[async] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())