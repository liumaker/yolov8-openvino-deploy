# YOLOv8 → ONNX → OpenVINO 端到端部署（CPU）

把 Ultralytics YOLOv8n 从 PyTorch 一路导出到 OpenVINO INT8，**前后处理全部自己实现**（letterbox / 解码 / 按类 NMS），
并用 NNCF 做 INT8 PTQ、用 COCO val2017 跑 mAP、用 `benchmark_app` + 多路异步流测性能。
每一步都有独立脚本和独立验收，数据可复现。

```
models/yolov8n.pt
  └─ scripts/export_onnx.py    → models/onnx/yolov8n_640_b1.onnx / _dyn.onnx
  └─ scripts/verify_onnx.py    → 与 PyTorch 原始输出逐值比对
  └─ scripts/export_ir.py      → models/ir/*_fp32.xml / *_fp16.xml
  └─ scripts/verify_ir.py      → ONNX vs IR 数值比对
  └─ scripts/infer_ov.py       → 自写 letterbox→infer→decode→NMS→映射回原图
  └─ scripts/quantize_nncf.py  → models/ir/*_int8*.xml（PTQ / accuracy-aware）
  └─ scripts/eval_map.py       → results/map_*.json（COCO mAP）
  └─ scripts/bench_openvino.py → results/bench_summary.json（benchmark_app）
  └─ scripts/async_stream.py   → results/stream_*.json（多路并发 + CPU）
```

配套文档：[`流程说明.md`](流程说明.md)（按执行顺序记录每一步的命令、产物、验收与结果）、
`models/说明.md`（产物与数值细节）、`models/排错.md`（LetterBox 精度事故复盘）。

---

## 1. 环境

| 项 | 版本 |
|---|---|
| CPU | i7-1165G7，8 逻辑核 / 4 物理核 |
| Python | 3.11.14（Anaconda base） |
| PyTorch | 2.5.1（CPU） |
| Ultralytics | 8.4.51 |
| OpenVINO | 2026.4.1 |
| NNCF | 3.4.0 |
| onnx / onnxruntime | 1.14.0 / 1.23.2 |
| pycocotools | 2.0.11 |

## 2. 流水线与验收结果

| 阶段 | 产物 | 验收 | 结果 |
|---|---|---|---|
| PyTorch → ONNX（静态） | `yolov8n_640_b1.onnx` 12.26 MiB | 原始输出 vs PyTorch | max\|Δ\| **0.00238**（框坐标，量级 ~637） |
| PyTorch → ONNX（动态） | `yolov8n_640_dyn.onnx` 12.47 MiB | 480×640→6300、batch2→16800 | PASS |
| ONNX → IR fp32 | `*_fp32.xml` + 12.12 MiB bin | vs ONNX 原始输出 | 相对误差 **3.2e-6** |
| ONNX → IR fp16 | `*_fp16.xml` + 6.06 MiB bin | vs ONNX 原始输出 | 相对误差 **5.0e-3** |
| 自写前后处理 | `src/preprocess.py` `src/postprocess.py` | vs Ultralytics 框 | 5/5 匹配，坐标差 **0.0001 px** |
| NNCF INT8 PTQ | `*_int8_aa2.xml` 3.86 MiB bin | COCO 500 张 mAP | **AP 0.3886 vs 0.3998（−1.12 点）** |
| benchmark_app | `results/bench_summary.json` | niter=150 | FP32 30.67 ms → INT8 **16.57 ms** |
| 多路异步 | `results/stream_*.json` | 3 次取中位数 | 4 路 INT8 **48.4 FPS** |

### 2.1 精度（COCO val2017 前 500 张，`conf=0.001 / NMS IoU=0.7`）

| 模型 | AP | AP50 | AP75 | 相对 FP32 |
|---|---|---|---|---|
| FP32 IR | **0.3998** | 0.5535 | 0.4390 | — |
| INT8 朴素 PTQ（performance） | 0.3410 | 0.5238 | 0.3859 | −5.9 点 / −14.7% |
| INT8 朴素 PTQ（mixed） | 0.3645 | 0.5360 | 0.4075 | −3.5 点 / −8.8% |
| INT8 mixed + 检测头保 FP32 | 0.3670 | 0.5414 | 0.4122 | −3.3 点 / −8.2% |
| **INT8 accuracy-aware（最终）** | **0.3886** | 0.5394 | 0.4244 | **−1.12 点 / −2.8%** |

校准集 300 张：coco128 的 128 张 + val2017 第 501–672 张；评测集 val2017 前 500 张，两者不重叠。
accuracy-aware 的验证池只用那 172 张 val2017 校准图（与评测集无交集），避免“拿考题调参”。

### 2.2 单模型性能（benchmark_app，`-api sync -hint latency` / `-api async -hint throughput`）

| 模型 | 延迟 ms（min） | 吞吐 FPS（latency 模式 / throughput 模式） | 模型体积 |
|---|---|---|---|
| FP32 | 30.67（26.38） | 31.7 / 32.2 | 12.12 MiB |
| FP16（仅权重压缩） | 34.84（27.91） | 28.0 / 31.9 | 6.06 MiB |
| INT8 朴素 | 12.28（10.63） | 76.5 / 74.0 | 3.14 MiB |
| **INT8 accuracy-aware** | **16.57（12.89）** | 57.4 / 61.5 | 3.86 MiB |

> CPU 上 FP16 只是权重压缩（计算仍转 fp32），所以**不比 FP32 快**，只省磁盘和内存。
> 真 FP16 加速要 Intel GPU。INT8 才是 CPU 上的有效手段。

### 2.3 端到端（含自写前后处理，单张 bus.jpg，20 次）

| 阶段 | FP32 | INT8 | 说明 |
|---|---|---|---|
| pre（letterbox） | 6.72 ms | 6.72 ms | 前后处理与精度无关 |
| infer | 44.92 ms | 17.66 ms | 纯前向 |
| post（decode+NMS） | 2.01 ms | 2.01 ms | |
| **e2e** | **56.93 ms（17.6 FPS）** | **26.39 ms（37.9 FPS）** | 整条链路 **2.16×** |

### 2.4 多路视频流（每配置 3 次 × 10 s，取中位数）

| 模型 | 路数 | 总 FPS（范围） | 进程 CPU% | 单帧 infer ms |
|---|---|---|---|---|
| FP32 | 1 | 18.9（9.5–19.4） | 328 | 49.9 |
| FP32 | 2 | 18.6（14.4–19.4） | 373 | 104.0 |
| FP32 | 4 | 25.5（22.1–26.3） | 641 | 152.0 |
| INT8 | 1 | 42.1（41.1–45.1） | 339 | 20.8 |
| INT8 | 2 | 45.1（27.7–45.3） | 369 | 41.5 |
| **INT8** | **4** | **48.4（48.1–55.6）** | 633 | 78.0 |

每路一个线程 + 独立 `InferRequest`，decode/NMS 与其它流的推理重叠。
4 路 INT8 ≈ 每路 12 FPS、总 48 FPS；8 路会过订阅（总 65 FPS 但单路掉到 8 FPS）。
8 逻辑核在 4 路时已被打满（进程 CPU ~630–640%），**瓶颈是 CPU 不是模型**。

---

## 3. 关键实现点

### 3.1 动态轴与输出解码

- 静态模型输出 `(1, 84, 8400)`：84 = 4 个框参数 + 80 类，**YOLOv8 无 objectness**。
  8400 = 80² + 40² + 20²（stride 8/16/32）。480×640 输入则对应 6300 = 60×80 + 30×40 + 15×20。
- 前 4 列**已经是输入图像素坐标**（anchor-free，不乘 stride），只需 `xywh→xyxy` 再除以 letterbox 的 ratio、减 pad。
- 置信度 = `max(cls_scores)`，按类别分别做 NMS（与 Ultralytics 行为一致）。

### 3.2 letterbox 必须与导出时一致

`ultralytics/engine/predictor.py:196` 的 `auto` 参数决定策略：

```python
auto = same_shapes and self.args.rect and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) ...))
```

同一张 810×1080 的图：`.pt` 与**动态** ONNX 走 `auto=True` → 喂 `640×480`（左右不补边）；
**静态** ONNX 走 `auto=False` → 喂 `640×640`（左右各补 80px 灰边 114）。像素不同，端到端结果自然差 647 px。
这不是模型错，是预处理不一致——见 `models/排错.md`。

本项目的 `src/preprocess.py: letterbox()` 固定 `640×640`、灰底 114、居中，
并返回 `LetterboxMeta(ratio, pad_w, pad_h, orig_w, orig_h)` 供 `to_original()` 反算。

### 3.3 NNCF 量化的三个坑

1. `ov.read_model` 在 2026.4.1 已不存在，必须 `ov.Core().read_model(path)`。
2. `ov.save_model(q_model, xml, compress_to_fp16=False)`——默认 `True` 会把 INT8 权重再压一次。
3. `quantize_with_accuracy_control` 的 `validation_fn` 拿到的**已经是 CompiledModel**，
   再 `compile_model()` 会报 `Path: '<CompiledModel: ...>' does not exist`，捕获 `RuntimeError` 后直接用。

### 3.4 benchmark_app 报告静默丢失

本机 2026.4.1 控制台只打到 `[Step 11/11] Dumping statistics report` 就退出（rc=0），表格不打印。
可用组合：`-report_folder <dir> -report_type average_counters -json_stats` → `<dir>/benchmark_report.json`。
注意 `-json_stats` 是**布尔开关**，传路径会报 `Boolean value expected`。

---

## 4. 复现

```powershell
python scripts/export_onnx.py --opset 12                 # 静态 ONNX
python scripts/export_onnx.py --dynamic                  # 动态 ONNX
python scripts/verify_onnx.py --onnx models/onnx/yolov8n_640_b1.onnx --probe
python scripts/export_ir.py --precision both             # fp32 + fp16 IR
python scripts/verify_ir.py
python scripts/infer_ov.py --ir models/ir/yolov8n_640_b1_fp32.xml

python scripts/prepare_coco.py --n-eval 500 --n-calib-extra 172
python scripts/eval_map.py --model models/ir/yolov8n_640_b1_fp32.xml --out results/map_fp32.json
python scripts/quantize_nncf.py --max-drop 0.01 --val-source calib --out models/ir/yolov8n_640_b1_int8_aa2.xml
python scripts/eval_map.py --model models/ir/yolov8n_640_b1_int8_aa2.xml --out results/map_int8_aa2.json

python scripts/bench_openvino.py
python scripts/async_stream.py --streams 4 --hint throughput --seconds 10
python scripts/agg_stream.py
```

> 数据：`data/coco/manifest.json` 记录 300 张校准 + 500 张评测的完整清单；
> val2017 图片与 `instances_val2017.json` 由 `scripts/prepare_coco.py` 自动下载。

---

## 5. 可写进简历的表述

- 完成 **YOLOv8n → ONNX → OpenVINO IR → INT8** 的 CPU 部署闭环，前后处理（letterbox / 解码 / 按类 NMS）全部自研，
  与 Ultralytics 参考实现逐框对齐（坐标误差 **0.0001 px**）。
- 用 **NNCF PTQ（300 张校准图）** 量化，并针对朴素量化掉点严重（−5.9 点）的问题改用 **accuracy-aware 逐层回退**：
  COCO val2017 500 张上 **mAP 0.3998 → 0.3886（−1.1 点）**，单帧延迟 **30.7 ms → 16.6 ms（−46%）**，吞吐 **32 → 57 FPS**。
- 实现 **多路视频流并发推理**（每路独立 `InferRequest`）：4 路 INT8 总吞吐 **48 FPS**，约为 FP32（25 FPS）的 **1.9×**，
  进程 CPU 占用 ~633%（8 逻辑核），定位到瓶颈为 CPU 算力而非模型。
- 踩通并记录 **LetterBox auto 策略、OpenVINO `save_model` 的 fp16 压缩语义、`benchmark_app` 报告静默丢失、
  NNCF `validation_fn` 收到 CompiledModel** 等部署细节。
