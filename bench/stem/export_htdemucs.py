#!/usr/bin/env python3
"""Export htdemucs (4 stems) to one self-contained ONNX file for the NS17a spike.

The STFT/ISTFT are the conv rewrite of adefossez/demucs PR #10 (dhunstack@6a96f8c), so the
browser host needs no FFT of its own. Writes fp32 and fp16 files plus their sizes and SHA-256
to <out>/htdemucs.json, and the largest |onnx - torch| on one random segment.
Licence of these weights: see README.md (not commercial_ok; spike only, never shipped).
"""
import argparse, hashlib, json, os, sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from onnxruntime.transformers.float16 import convert_float_to_float16  # handles Cast nodes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = get_model("htdemucs")
    core = model if isinstance(model, HTDemucs) else model.models[0]
    core.eval()
    core.onnx_exportable = True
    torch.backends.mha.set_fastpath_enabled(False)  # the fused MHA kernel has no ONNX symbolic
    seg = int(core.segment * core.samplerate)
    x = torch.randn(1, 2, seg)
    fp32 = os.path.join(a.out, "htdemucs.onnx")
    with torch.no_grad():
        torch.onnx.export(core, (x,), fp32, export_params=True, opset_version=17,
                          do_constant_folding=True, input_names=["mix"], output_names=["stems"])
        ref = core(x).numpy()
    got = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).run(None, {"mix": x.numpy()})[0]
    fp16 = os.path.join(a.out, "htdemucs.fp16.onnx")
    m16 = convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
    onnx.save(m16, fp16)
    got16 = ort.InferenceSession(fp16, providers=["CPUExecutionProvider"]).run(None, {"mix": x.numpy()})[0]
    params = sum(p.numel() for p in core.parameters())
    info = {
        "model": "htdemucs", "sources": list(core.sources), "samplerate": core.samplerate,
        "segment_samples": seg, "segment_seconds": seg / core.samplerate, "params": params,
        "input": [1, 2, seg], "output": list(ref.shape), "opset": 17,
        "files": [{"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)} for p in (fp32, fp16)],
        "max_abs_err_fp32": float(np.abs(got - ref).max()),
        "max_abs_err_fp16": float(np.abs(got16 - ref).max()),
        "ref_abs_max": float(np.abs(ref).max()),
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
    }
    with open(os.path.join(a.out, "htdemucs.json"), "w") as f:
        json.dump(info, f, indent=1)
    json.dump(info, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
