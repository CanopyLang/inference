#!/usr/bin/env python3
"""Compare each bench result's output sample with ORT CPU on the same seeded input.

usage: check-outputs.py MODELS_DIR results.jsonl  -> one line per case with the relative error.
"""
import json, sys

import numpy as np
import onnxruntime as ort

SHAPES = {"mix": (1, 2, 343980), "mag": (2, 1, 512, 1024)}


def lcg(n):
    out, s = np.empty(n, np.float32), 12345
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = s / 4294967296 * 2 - 1
    return out


def main():
    models, path = sys.argv[1], sys.argv[2]
    refs = {}
    for line in open(path):
        r = json.loads(line)
        if not r.get("ok") or "outSample" not in r:
            continue
        base = r["model"].replace(".fp16", "")
        if base not in refs:
            s = ort.InferenceSession(f"{models}/{base}", providers=["CPUExecutionProvider"])
            name = s.get_inputs()[0].name
            x = lcg(int(np.prod(SHAPES[name]))).reshape(SHAPES[name])
            if name == "mag":
                x = np.abs(x)
            y = s.run(None, {name: x})[0].ravel()
            step = len(y) // 64
            refs[base] = (y[::step][:64], float(np.abs(y).mean()))
        ref, mean = refs[base]
        got = np.array(r["outSample"], np.float32)
        rel = float(np.abs(got - ref).mean() / max(1e-12, np.abs(ref).mean()))
        print(json.dumps({"model": r["model"], "ep": r["ep"], "threads": r["threads"], "power": r.get("power"),
                          "rel_err_sample": rel, "abs_mean": r["outAbsMean"], "ref_abs_mean": mean}))


if __name__ == "__main__":
    main()
