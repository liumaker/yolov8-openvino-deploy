"""打印 yolov8n.pt -> ONNX -> IR(fp32/fp16) -> INT8 各阶段权重的存储形态对比。

用法：python scripts/weights_evolution.py
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

import onnx  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402
from onnx import TensorProto  # noqa: E402

DTYPE_BYTES = {
    TensorProto.FLOAT: 4, TensorProto.FLOAT16: 2, TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1, TensorProto.UINT8: 1, TensorProto.INT32: 4,
    TensorProto.INT64: 8, TensorProto.BOOL: 1,
}


def dtype_name(t: int) -> str:
    return TensorProto.DataType.Name(t)


def inspect_pt(path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    dtypes: dict[str, int] = {}
    elems = 0
    for p in model.parameters():
        dtypes[str(p.dtype).replace("torch.", "")] = dtypes.get(str(p.dtype).replace("torch.", ""), 0) + p.numel()
        elems += p.numel()
    print(f"[pt ] {path.name}: 文件 {path.stat().st_size / 2**20:.2f} MiB")
    print(f"      顶层键: {sorted(ckpt.keys())}")
    nbytes = elems * 2 if "float16" in dtypes else elems * 4
    print(f"      参数量 {elems:,}，dtype 分布 {dtypes}")
    print(f"      纯权重字节数 {nbytes:,} B = {nbytes / 2**20:.2f} MiB"
          f"（文件里剩下的是 pickle 结构、BN buffer、EMA 等）")


def inspect_onnx(path: Path) -> None:
    m = onnx.load(str(path))
    inits = m.graph.initializer
    elems = 0
    total = 0
    for i in inits:
        n = 1
        for d in i.dims:
            n *= d
        elems += n
        total += n * DTYPE_BYTES.get(i.data_type, 0)
    print(f"[onnx] {path.name}: 文件 {path.stat().st_size / 2**20:.2f} MiB，opset {m.opset_import[0].version}")
    print(f"      输入/输出: {[f'{d.name}{list(d.type.tensor_type.shape.dim)}' for d in list(m.graph.input) + list(m.graph.output)]}")
    print(f"      initializer {len(inits)} 个，共 {elems:,} 元素，按 float32 计 {total / 2**20:.2f} MiB")
    kinds = {dtype_name(i.data_type) for i in inits}
    print(f"      initializer 数据类型: {kinds}")
    for k in kinds:
        sub = [i for i in inits if dtype_name(i.data_type) == k]
        n = sum(int(__import__('math').prod(i.dims)) for i in sub)
        print(f"        - {k}: {len(sub)} 个，{n:,} 元素")


def inspect_ir(xml: Path) -> None:
    core = ov.Core()
    m = core.read_model(str(xml))
    ops: dict[str, int] = {}
    for op in m.get_ops():
        ops[op.get_type_name()] = ops.get(op.get_type_name(), 0) + 1
    bin_size = xml.with_suffix(".bin").stat().st_size
    consts = ops.get("Constant", 0)
    print(f"[ir ] {xml.name}: xml {xml.stat().st_size / 2**10:.0f} KiB，bin {bin_size / 2**20:.2f} MiB，"
          f"Constant {consts} 个，节点 {len(m.get_ops())}")
    print(f"      {ops}")
    for elem_bytes in (4, 2, 1):
        if bin_size % elem_bytes == 0:
            print(f"      若每个权重 {elem_bytes} B → 约 {bin_size // elem_bytes:,} 个元素")


if __name__ == "__main__":
    print("=" * 78)
    inspect_pt(ROOT / "models" / "yolov8n.pt")
    print("=" * 78)
    inspect_onnx(ROOT / "models" / "onnx" / "yolov8n_640_b1.onnx")
    print("=" * 78)
    inspect_ir(ROOT / "models" / "ir" / "yolov8n_640_b1_fp32.xml")
    print("=" * 78)
    inspect_ir(ROOT / "models" / "ir" / "yolov8n_640_b1_fp16.xml")
    print("=" * 78)
    inspect_ir(ROOT / "models" / "ir" / "yolov8n_640_b1_int8_aa2.xml")
