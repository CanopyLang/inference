#!/usr/bin/env python3
"""Lay out a stem model as a Canopy mount's `_/` (edsite docs/canopy-mount-contract.md §1, §6): the model
cut into chunks named by sha256 (`<sha>.part`), the chunk manifest (`<sha>.json`), and the host's model
descriptor with a reference chunk checked by ORT CPU, which stem-worker.mjs compares each backend against.

usage: make-mount.py MODEL.onnx MODEL.json OUT_DIR [--chunk-bytes N]   (box only; MODEL.json from export_spleeter.py)
"""
import argparse, hashlib, json, os

import numpy as np
import onnxruntime as ort

# 1e-3 on a mask in [0, 1]: four orders above the fp32 agreement the spike measured on WebGPU and WASM
# (1.3e-7), far below the broken backends it found (rel err 5.9, all-zero). README §3.
REF_MAX_ABS_ERR = 1e-3
REF_SEED = 12345


def lcg_abs(n, seed):
    out = np.empty(n, dtype=np.float32)
    s = seed
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = abs(s / 4294967296 * 2 - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("info"); ap.add_argument("out")
    ap.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)  # tools.chunk_max_bytes's first value
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(a.info))
    data = open(a.model, "rb").read()
    chunks = []
    for at in range(0, len(data), a.chunk_bytes):
        c = data[at:at + a.chunk_bytes]
        h = hashlib.sha256(c).hexdigest()
        open(os.path.join(a.out, f"{h}.part"), "wb").write(c)
        chunks.append({"sha256": h, "bytes": len(c)})
    manifest = json.dumps({"name": os.path.basename(a.model), "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest(), "chunks": chunks}, indent=1).encode()
    msha = hashlib.sha256(manifest).hexdigest()
    open(os.path.join(a.out, f"{msha}.json"), "wb").write(manifest)

    T, F = info["chunk_frames"], info["bins"]
    x = lcg_abs(2 * T * F, REF_SEED).reshape(2, 1, T, F)
    y = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"]).run(None, {"mag": x})[0].ravel()
    rng = np.random.default_rng(REF_SEED)
    idx = np.unique(np.concatenate([np.linspace(0, y.size - 1, 256).astype(np.int64), rng.integers(0, y.size, 256)]))
    desc = {
        "name": info["model"], "sources": info["sources"], "sampleRate": info["samplerate"], "frame": info["frame"],
        "hop": info["hop"], "bins": F, "chunkFrames": T, "input": "mag", "output": "masks",
        "model": {"manifest": f"{msha}.json", "manifestSha256": msha},
        "reference": {"seed": REF_SEED, "indices": idx.tolist(), "values": [float(v) for v in y[idx]], "maxAbsErr": REF_MAX_ABS_ERR},
    }
    name = f"{os.path.splitext(os.path.basename(a.model))[0]}.stem.json"
    json.dump(desc, open(os.path.join(a.out, name), "w"))
    print(json.dumps({"descriptor": name, "manifest": msha, "chunks": len(chunks), "bytes": len(data)}))


if __name__ == "__main__":
    main()
