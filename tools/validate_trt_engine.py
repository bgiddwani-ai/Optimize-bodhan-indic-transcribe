#!/usr/bin/env python3
"""Deserialize a TensorRT engine and run its minimum-profile shape once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch


TORCH_DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_bytes = Path(args.engine).read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context")

    tensors: list[dict] = []
    buffers: dict[str, torch.Tensor] = {}
    minimum_feature_length = 1
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        entry = {
            "name": name,
            "mode": str(mode),
            "dtype": str(dtype),
            "engine_shape": list(engine.get_tensor_shape(name)),
        }
        if mode == trt.TensorIOMode.INPUT:
            min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(
                name, 0
            )
            entry["profile"] = {
                "min": list(min_shape),
                "opt": list(opt_shape),
                "max": list(max_shape),
            }
            if len(min_shape) > 1:
                minimum_feature_length = max(minimum_feature_length, int(min_shape[-1]))
            if not context.set_input_shape(name, min_shape):
                raise RuntimeError(f"Could not set minimum input shape for {name}")
        tensors.append(entry)

    if not context.infer_shapes() == []:
        raise RuntimeError(f"Unspecified tensors after shape inference: {context.infer_shapes()}")

    for entry in tensors:
        name = entry["name"]
        shape = tuple(context.get_tensor_shape(name))
        entry["runtime_shape"] = list(shape)
        trt_dtype = engine.get_tensor_dtype(name)
        if trt_dtype not in TORCH_DTYPES:
            raise RuntimeError(f"Unsupported validation dtype {trt_dtype} for {name}")
        dtype = TORCH_DTYPES[trt_dtype]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if dtype in (torch.int32, torch.int64):
                tensor = torch.full(
                    shape, minimum_feature_length, dtype=dtype, device="cuda"
                )
            else:
                tensor = torch.randn(shape, dtype=dtype, device="cuda")
        else:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
        buffers[name] = tensor
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind tensor {name}")

    stream = torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT execution returned false")
    torch.cuda.synchronize()

    outputs = []
    for entry in tensors:
        name = entry["name"]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            tensor = buffers[name]
            finite = bool(torch.isfinite(tensor.float()).all().item())
            outputs.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "finite": finite,
                    "mean_abs": float(tensor.float().abs().mean().item()),
                }
            )
            if not finite:
                raise RuntimeError(f"Non-finite output from {name}")

    report = {
        "engine": str(Path(args.engine)),
        "engine_bytes": len(engine_bytes),
        "num_profiles": engine.num_optimization_profiles,
        "tensors": tensors,
        "minimum_profile_execution": "passed",
        "outputs": outputs,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
