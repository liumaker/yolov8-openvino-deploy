"""Run the official benchmark_app over FP32 / FP16 / INT8 IRs and build a comparison table.

benchmark_app 2026.4 on this machine prints "Step 11/11 Dumping statistics report" but never
the console table (silent report bug), so this wrapper passes -report_folder and parses
benchmark_report.json instead.

Two passes per model:
  * throughput: -api async -hint throughput (all CPU threads, 4 requests)
  * latency:    -api sync  -hint latency    (one frame at a time)

Usage:
  python scripts/bench_openvino.py
  python scripts/bench_openvino.py --niter 200 --models fp32=models/ir/....xml
"""
from __future__ import annotations

import argparse  # 解析命令行参数（--models / --device / --niter / --out）
import json  # 读写 benchmark_report.json 与最终的汇总结果 JSON
import shutil  # 删除旧的报告目录
import subprocess  # 拉起外部进程 benchmark_app.exe
import sys  # 获取当前 Python 解释器路径、退出码等
import time  # 统计每次 benchmark 的墙钟耗时

import psutil  # 采集系统 CPU 占用率与进程 CPU 占用率

# 项目根目录：由本文件路径上溯一级得到，保证脚本可从任意工作目录运行
ROOT = Path(__file__).resolve().parents[1]
# 定位 benchmark_app.exe：优先找与当前 Python 同目录的，兜底找 Scripts/ 下的
BENCH = Path(sys.executable).with_name("benchmark_app.exe")
if not BENCH.exists():
    BENCH = Path(sys.executable).with_name("Scripts") / "benchmark_app.exe"

# 存放每次 benchmark 生成的报告（benchmark_report.json + 日志）的目录
REPORTS = ROOT / "results" / "bench_reports"

# 默认的基准测试模型列表：(名称, IR 模型路径)
# fp32 / fp16 用于对照原始精度，int8_aa / int8_naive 用于对比两种量化策略的效果
DEFAULTS = [
    ("fp32", ROOT / "models" / "ir" / "yolov8n_640_b1_fp32.xml"),
    ("fp16", ROOT / "models" / "ir" / "yolov8n_640_b1_fp16.xml"),
    ("int8_aa", ROOT / "models" / "ir" / "yolov8n_640_b1_int8_aa2.xml"),
    ("int8_naive", ROOT / "models" / "ir" / "yolov8n_640_b1_int8.xml"),
]

# 两种测试模式及对应的 benchmark_app 参数：
# throughput = 异步 + 吞吐量优先，用满全部 CPU 线程并发处理，衡量单位时间处理帧数
# latency    = 同步 + 延迟优先，一次只推理一帧，衡量单帧端到端耗时
PASSES = [
    ("throughput", ["-api", "async", "-hint", "throughput"]),
    ("latency", ["-api", "sync", "-hint", "latency"]),
]


def parse_report(folder: Path) -> dict:
    """解析 benchmark_app 生成的 benchmark_report.json。

    因为本机 benchmark_app 控制台不打印结果表，只能从 JSON 报告里取数据。
    - folder: 存放 benchmark_report.json 的目录
    - 返回: {"字段名": 数值}，数值尽量转成 float 便于后续计算/输出
    """
    f = folder / "benchmark_report.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    # execution_results 里是 "指标名 -> 值" 的映射（如 throughput、avg latency 等）
    er = data.get("execution_results", {})
    out = {}
    for k, v in er.items():
        try:
            out[k] = float(v)  # 能转成 float 的（如耗时、FPS）转成数值
        except (TypeError, ValueError):
            out[k] = v  # 转不了的（如字符串）原样保留
    return out


def run_one(name: str, model: Path, mode: str, extra: list[str], args) -> dict:
    """对单个 (模型, 测试模式) 组合执行一次完整 benchmark。

    步骤：
      1. 清空并重建该组合的报告目录
      2. 用 subprocess 异步拉起 benchmark_app，实时监控系统/进程 CPU 占用
      3. 运行结束后读取 stdout 日志并解析 JSON 报告
      4. 汇总成一行结果字典返回

    参数:
      name  - 模型名称（如 fp32），用于命名目录/日志
      model - IR 模型文件路径
      mode  - 测试模式名称（throughput / latency），用于命名目录/日志
      extra - 该模式额外的 benchmark_app 参数
      args  - 脚本自身的命令行参数（device / niter / niter 等）
    """
    # --- 1. 准备输出目录 ---
    folder = REPORTS / f"{name}_{mode}"
    if folder.exists():
        shutil.rmtree(folder)  # 清掉上一次的遗留报告，避免解析到旧数据
    folder.mkdir(parents=True, exist_ok=True)

    # --- 2. 组装并启动外部进程 ---
    # -report_type average_counters 会输出每次推理的平均计数（含各层耗时）
    # -json_stats 让报告以 JSON 形式落盘，便于解析
    cmd = [str(BENCH), "-m", str(model), "-d", args.device, "-niter", str(args.niter),
           *extra, "-report_folder", str(folder), "-report_type", "average_counters", "-json_stats"]

    t0 = time.time()  # 记录墙钟起点
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    # 对子进程建立 psutil 句柄，用于统计它自身占用的 CPU 百分比
    try:
        proc = psutil.Process(p.pid)
        proc.cpu_percent(None)  # 首次调用只作为基准点，不返回值
    except psutil.Error:
        proc = None
    sys_cpu: list[float] = []   # 系统整体 CPU 占用采样序列
    proc_cpu: list[float] = []  # 子进程 CPU 占用采样序列

    # --- 3. 轮询等待进程结束，期间定时采样 CPU 占用 ---
    while p.poll() is None:
        time.sleep(0.25)
        sys_cpu.append(psutil.cpu_percent(None))
        if proc is not None:
            try:
                proc_cpu.append(proc.cpu_percent(None))
            except psutil.Error:
                pass  # 进程已退出但没来得及结束循环时忽略采样失败

    # --- 4. 收尾：读取 stdout、计算耗时、落盘日志、解析报告 ---
    out = p.stdout.read() if p.stdout else ""
    dt = time.time() - t0  # 本次 benchmark 的墙钟耗时（秒）
    (REPORTS / f"{name}_{mode}.log").write_text(out, encoding="utf-8", errors="replace")

    er = parse_report(folder)
    return {
        "name": name,
        "mode": mode,
        "model": str(model),
        "device": args.device,
        "niter": args.niter,
        "throughput_fps": er.get("throughput"),                    # 吞吐量（帧/秒）
        "latency_ms": er.get("avg latency", er.get("latency (ms)")),  # 平均延迟（毫秒），兼容两种字段名
        "latency_min_ms": er.get("min latency"),                   # 最小延迟
        "latency_max_ms": er.get("max latency"),                   # 最大延迟
        "first_infer_ms": er.get("first inference time (ms)"),     # 首次推理耗时（含模型预热开销）
        "compile_ms": er.get("compile model time (ms)"),           # 模型编译耗时
        "wall_s": round(dt, 1),                                    # 本次运行墙钟耗时
        "cpu_sys_mean_pct": round(sum(sys_cpu) / len(sys_cpu), 1) if sys_cpu else None,   # 系统 CPU 平均占用
        "cpu_proc_mean_pct": round(sum(proc_cpu) / len(proc_cpu), 1) if proc_cpu else None,  # 进程 CPU 平均占用
        "returncode": p.returncode,                                # 进程退出码，非 0 说明失败
    }


def main() -> int:
    """主流程：解析参数 -> 遍历所有模型和模式跑 benchmark -> 输出对比表。"""
    # --- 命令行参数定义 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="name=path pairs; default fp32/fp16/int8_aa/int8_naive")
    ap.add_argument("--device", default="CPU")          # 推理设备，默认 CPU
    ap.add_argument("--niter", type=int, default=150)   # 每次 benchmark 的推理迭代次数
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bench_summary.json")  # 汇总输出路径
    args = ap.parse_args()

    # --- 检查 benchmark_app 是否存在，不存在直接报错退出 ---
    if not BENCH.exists():
        raise SystemExit(f"benchmark_app not found: {BENCH}")

    # --- 确定要测试的模型列表：默认四件套，或解析用户传入的 name=path ---
    models = DEFAULTS
    if args.models:
        models = []
        for spec in args.models:
            n, p = spec.split("=", 1)   # 按第一个 "=" 拆成 (名称, 路径)
            models.append((n, Path(p)))

    # --- 遍历所有模型 x 所有模式执行 benchmark ---
    rows = []
    for name, path in models:
        for mode, extra in PASSES:
            print(f"[bench] {name} / {mode}: {Path(path).name} (niter={args.niter})", flush=True)
            row = run_one(name, Path(path), mode, extra, args)
            rows.append(row)
            # 打印本行关键指标：吞吐量、平均/最小/最大延迟、进程/系统 CPU 占用
            print(f"        tput={row['throughput_fps']} FPS  lat={row['latency_ms']} ms "
                  f"(min {row['latency_min_ms']} / max {row['latency_max_ms']})  "
                  f"cpu_proc={row['cpu_proc_mean_pct']}%  cpu_sys={row['cpu_sys_mean_pct']}%", flush=True)
            if row["returncode"] != 0 or row["throughput_fps"] is None:
                # 退出码非 0 或没取到吞吐量说明本次失败，显式提示
                print("        FAILED rc=" + str(row["returncode"]), flush=True)

    # --- 全量结果写入 JSON，方便后续程序化分析 ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # --- 控制台打印 Markdown 对比表，一眼对比各精度性能 ---
    print("\n| model | mode | throughput (FPS) | latency (ms) | min/max (ms) | cpu proc (%) | cpu sys (%) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['mode']} | {r['throughput_fps']} | {r['latency_ms']} | "
              f"{r['latency_min_ms']} / {r['latency_max_ms']} | {r['cpu_proc_mean_pct']} | {r['cpu_sys_mean_pct']} |")
    print(f"\n[bench] wrote {args.out} and {REPORTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())