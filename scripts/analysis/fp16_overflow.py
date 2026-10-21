#!/usr/bin/env python3
"""
Does an fp16 export overflow on backends that run its normalization ops in fp16?

Background (docs/model-compatibility.md#fp16-overflow-on-webgpu): the fp16 / q4f16 exports of
SmolLM2 and Qwen2.5 compute RMSNorm as separate fp16 ops (Pow -> ReduceMean -> Add -> Sqrt ->
Div -> Mul). ONNX Runtime's SimplifiedLayerNormFusion only runs for the CPU and CUDA
execution providers, so on WASM (CPU EP) the pattern becomes one fp32-accumulating kernel,
while the WebGPU EP executes it as-is in fp16. If any |x| at a norm input exceeds sqrt(65504)
~= 255.9, x**2 is +inf in fp16, the mean is inf, 1/sqrt(inf) = 0 and the normalized hidden
state is exactly 0.

This script measures that with the fp32 graph on CPU:
  1. the largest |x| at every Pow/LayerNorm input (the activations the fp16 graph would see);
  2. an emulation of the fp16 overflow: every Pow output above 65504 is replaced by +inf
     (exactly what an fp16 store does), then the logits are compared with the reference.

  python3 -m venv .venv && .venv/bin/pip install onnx onnxruntime numpy
  .venv/bin/python scripts/analysis/fp16_overflow.py \
      tests/fixtures/models/hf/onnx-community/SmolLM2-135M-Instruct-ONNX/<rev>/onnx/model.onnx \
      '[[504, 3575, 282, 4649, 314], [0]]'

Token-ID lists are prompts (e.g. "The capital of France is" for SmolLM2 = 504 3575 282 4649 314).
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

FP16_MAX = 65504.0


def feed(session, ids):
    n = len(ids)
    out = {}
    for i in session.get_inputs():
        if i.name == "input_ids":
            out[i.name] = np.array([ids], np.int64)
        elif i.name == "attention_mask":
            out[i.name] = np.ones((1, n), np.int64)
        elif i.name == "position_ids":
            out[i.name] = np.arange(n, dtype=np.int64)[None]
        elif i.name == "num_logits_to_keep":
            out[i.name] = np.array(1, np.int64)
        else:  # empty past key/values (or recurrent states)
            shape = [1 if d == "batch_size" else (0 if isinstance(d, str) else d) for d in i.shape]
            out[i.name] = np.zeros(shape, np.float32)
    return out


def softmax(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def save_next_to(src, model, name):
    # Keeps relative external-data references valid and avoids the 2 GB protobuf limit.
    path = os.path.join(os.path.dirname(src), name)
    onnx.save(model, path)
    return path


def main(src, prompts):
    # 1. activation magnitudes at norm inputs
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    norm_ops = {"Pow", "SimplifiedLayerNormalization", "LayerNormalization",
                "SkipSimplifiedLayerNormalization", "SkipLayerNormalization"}
    names = []
    for n in g.node:
        if n.op_type in norm_ops:
            for x in n.input[:2] if n.op_type.startswith("Skip") else n.input[:1]:
                if x and x not in names:
                    names.append(x)
    have = {o.name for o in g.output}
    for x in names:
        if x not in have:
            g.output.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, None))
    probe = save_next_to(src, m, "_probe_norm_inputs.onnx")
    try:
        s = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
    finally:
        os.remove(probe)
    for ids in prompts:
        vals = s.run(names, feed(s, ids))
        mx = sorted(((float(np.abs(v).max()), x) for x, v in zip(names, vals)), reverse=True)
        over = sum(a > 255.9 for a, _ in mx)
        print(f"prompt of {len(ids)} tokens: largest |x| at a norm input = {mx[0][0]:.1f} "
              f"({mx[0][1]}); {over}/{len(mx)} norm inputs have |x| > 255.9 (x^2 overflows fp16)")

    # 2. emulate the fp16 overflow of Pow outputs
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    g.initializer.append(helper.make_tensor("_f16max", TensorProto.FLOAT, [], [FP16_MAX]))
    g.initializer.append(helper.make_tensor("_inf", TensorProto.FLOAT, [], [float("inf")]))
    nodes = []
    for n in g.node:
        nodes.append(n)
        if n.op_type == "Pow":
            y = n.output[0]
            n.output[0] = y + "_raw"
            nodes += [helper.make_node("Greater", [y + "_raw", "_f16max"], [y + "_ovf"]),
                      helper.make_node("Where", [y + "_ovf", "_inf", y + "_raw"], [y])]
    del g.node[:]
    g.node.extend(nodes)
    if not any(n.op_type == "Pow" for n in nodes):
        print("No decomposed Pow-based norms in this graph: the fused norm kernels accumulate in fp32.")
        return
    probe = save_next_to(src, m, "_probe_fp16_overflow.onnx")
    try:
        emu = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
    finally:
        os.remove(probe)
    ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    for ids in prompts:
        a = ref.run(["logits"], feed(ref, ids))[0][0, -1]
        b = emu.run(["logits"], feed(emu, ids))[0][0, -1]
        if not np.isfinite(b).all() or b.max() - b.min() < 1e-6:
            print(f"prompt of {len(ids)} tokens: DEGENERATE logits under fp16 overflow "
                  f"(min {np.nanmin(b)}, max {np.nanmax(b)}) -> uniform distribution")
            continue
        p, q = softmax(a), softmax(b)
        kl = float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())
        print(f"prompt of {len(ids)} tokens: KL(reference || fp16-overflow) = {kl:.3f} nats; "
              f"top-1 {int(a.argmax())} -> {int(b.argmax())}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], json.loads(sys.argv[2]))
