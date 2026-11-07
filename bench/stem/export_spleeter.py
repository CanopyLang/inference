#!/usr/bin/env python3
"""Export Spleeter 4stems (deezer/spleeter v1.4.0 release, MIT code and weights) to ONNX.

One file holds the four U-Nets and spleeter's ratio mask: input the stereo magnitude
[2, n, 512, 1024] (frame 4096, hop 1024, first 1024 bins), output the masks [4, 2, n, 512, 1024]
in the order of 4stems.json's instrument_list. The STFT/ISTFT stay in the host (JS), as in
spleeter itself. The U-Net is sherpa-onnx's scripts/spleeter/unet.py (Apache-2.0), vendored.
"""
import argparse, hashlib, json, os, re, sys, tarfile, urllib.request

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16  # handles Cast nodes

from unet import UNet

URL = "https://github.com/deezer/spleeter/releases/download/v1.4.0/4stems.tar.gz"
# deezer/spleeter configs/4stems/base_config.json (instrument_list); the graph builds the
# U-Nets in this order, so the TF variable suffixes number them in it.
INSTRUMENTS = ["vocals", "drums", "bass", "other"]
EPS = 1e-10  # spleeter.model.EPSILON


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tfname(base, i):
    return base if i == 0 else f"{base}_{i}"


class Spleeter4(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    def forward(self, mag):
        est = torch.stack([n(mag) for n in self.nets])  # [4, 2, n, 512, 1024]
        sq = est * est
        return (sq + EPS / 4) / (sq.sum(0, keepdim=True) + EPS)


def load_nets(ckpt):
    r = tf.train.load_checkpoint(ckpt)
    names = set(r.get_variable_to_shape_map())
    counts = {k: len({n.split("/")[0] for n in names if re.fullmatch(k + r"(_\d+)?/.*", n)})
              for k in ("conv2d", "conv2d_transpose", "batch_normalization")}
    assert counts == {"conv2d": 28, "conv2d_transpose": 24, "batch_normalization": 48}, counts
    g = lambda n: torch.from_numpy(r.get_tensor(n))
    nets = []
    for k in range(4):
        net = UNet().eval()
        sd = net.state_dict()
        convs = ["conv", "conv1", "conv2", "conv3", "conv4", "conv5", "up7"]
        for j, name in enumerate(convs):
            t = tfname("conv2d", 7 * k + j)
            sd[f"{name}.weight"] = g(f"{t}/kernel").permute(3, 2, 0, 1)
            sd[f"{name}.bias"] = g(f"{t}/bias")
        for j in range(6):
            t = tfname("conv2d_transpose", 6 * k + j)
            sd[f"up{j + 1}.weight"] = g(f"{t}/kernel").permute(3, 2, 0, 1)
            sd[f"up{j + 1}.bias"] = g(f"{t}/bias")
        # 12 BNs per net; index 5 normalises the last encoder conv, which the decoder never reads.
        bns = ["bn", "bn1", "bn2", "bn3", "bn4", None, "bn5", "bn6", "bn7", "bn8", "bn9", "bn10"]
        for j, name in enumerate(bns):
            if name is None:
                continue
            t = tfname("batch_normalization", 12 * k + j)
            for ours, theirs in (("weight", "gamma"), ("bias", "beta"),
                                 ("running_mean", "moving_mean"), ("running_var", "moving_variance")):
                sd[f"{name}.{ours}"] = g(f"{t}/{theirs}")
        net.load_state_dict(sd)
        nets.append(net)
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tgz = os.path.join(a.out, "4stems.tar.gz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(URL, tgz)
    d = os.path.join(a.out, "4stems")
    if not os.path.isdir(d):
        with tarfile.open(tgz) as t:
            t.extractall(d)
    model = Spleeter4(load_nets(tf.train.latest_checkpoint(d) or os.path.join(d, "model"))).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.rand(2, 1, 512, 1024)
    fp32 = os.path.join(a.out, "spleeter4.onnx")
    with torch.no_grad():
        torch.onnx.export(model, (x,), fp32, input_names=["mag"], output_names=["masks"],
                          dynamic_axes={"mag": {1: "n"}, "masks": {2: "n"}}, opset_version=17)
        ref = model(x).numpy()
    got = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).run(None, {"mag": x.numpy()})[0]
    fp16 = os.path.join(a.out, "spleeter4.fp16.onnx")
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), fp16)
    got16 = ort.InferenceSession(fp16, providers=["CPUExecutionProvider"]).run(None, {"mag": x.numpy()})[0]
    info = {
        "model": "spleeter-4stems", "source": URL, "tarball_sha256": sha256(tgz), "sources": INSTRUMENTS,
        "samplerate": 44100, "frame": 4096, "hop": 1024, "bins": 1024, "chunk_frames": 512,
        "chunk_seconds": 512 * 1024 / 44100, "params": params, "input": [2, "n", 512, 1024],
        "output": [4, 2, "n", 512, 1024], "opset": 17,
        "files": [{"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)} for p in (fp32, fp16)],
        "max_abs_err_fp32": float(np.abs(got - ref).max()), "max_abs_err_fp16": float(np.abs(got16 - ref).max()),
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "tensorflow": tf.__version__,
    }
    with open(os.path.join(a.out, "spleeter4.json"), "w") as f:
        json.dump(info, f, indent=1)
    json.dump(info, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
