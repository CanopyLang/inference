#!/usr/bin/env python3
"""Spleeter's own pipeline in TensorFlow (tf.signal.stft/inverse_stft as spleeter's EstimatorSpecBuilder
calls them) with the exported masks on ORT CPU, compared with the stems the browser host wrote: SNR and
max abs difference per stem. This checks the host's STFT, masks and ISTFT, not the model.

usage: ref-separate.py MODEL.onnx TRACK.wav STEM_DIR PREFIX   (STEM_DIR holds PREFIX-<source>.wav)
"""
import argparse, json, os, wave

import numpy as np
import onnxruntime as ort
import tensorflow as tf

SOURCES = ["vocals", "drums", "bass", "other"]
N, HOP, F, T = 4096, 1024, 1024, 512


def read(path, full_scale=32767):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())
    return x.astype(np.float32) / full_scale


def write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def snr(ref, err):
    return float(10 * np.log10((ref ** 2).sum() / max((err ** 2).sum(), 1e-20)))


def hann(n, dtype):
    return tf.signal.hann_window(n, periodic=True, dtype=dtype)


WIN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def stft64(x):
    wav = np.concatenate([np.zeros((N, 2)), x.astype(np.float64)])
    frames = -(-len(wav) // HOP)
    wav = np.concatenate([wav, np.zeros((frames * HOP + N - len(wav), 2))])
    idx = np.arange(frames)[:, None] * HOP + np.arange(N)[None, :]
    return np.fft.rfft(wav[idx] * WIN[None, :, None], axis=1)  # [frames, 2049, 2]


def istft64(spec):
    frames = spec.shape[0]
    out = np.zeros(((frames - 1) * HOP + N, 2))
    y = np.fft.irfft(spec, n=N, axis=1) * WIN[None, :, None] * (2 / 3)
    for f in range(frames):
        out[f * HOP:f * HOP + N] += y[f]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("track"); ap.add_argument("stems"); ap.add_argument("prefix")
    ap.add_argument("--write-ref", help="also write the reference stems as ref-<source>.wav into this directory")
    ap.add_argument("--segments", action="store_true", help="also report SNR per model chunk (512 frames)")
    ap.add_argument("--float64", action="store_true",
                    help="do the STFT/ISTFT in float64 numpy (the host's arithmetic) instead of TF's float32")
    ap.add_argument("--ort-opt-off", action="store_true",
                    help="run ORT CPU with graph optimisation off: the model's own noise floor between two correct runs")
    a = ap.parse_args()
    x = read(a.track, 32768)  # as Web Audio's decodeAudioData scales 16-bit PCM; the host writes stems at 32767
    if a.float64:
        stft = stft64(x)
    else:
        wav = tf.concat([tf.zeros((N, 2)), x], 0)
        stft = tf.transpose(tf.signal.stft(tf.transpose(wav), N, HOP, window_fn=hann, pad_end=True), [1, 2, 0]).numpy()  # [frames, 2049, 2]
    frames = stft.shape[0]
    mag = np.abs(stft)[:, :F, :].astype(np.float32)
    pad = (-frames) % T
    mag = np.pad(mag, ((0, pad), (0, 0), (0, 0)))
    seg = mag.reshape(-1, T, F, 2).transpose(3, 0, 1, 2)  # [2, n, T, F]
    so = ort.SessionOptions()
    if a.ort_opt_off:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    masks = np.concatenate([sess.run(None, {"mag": seg[:, i:i + 1]})[0] for i in range(seg.shape[1])], axis=2)  # [4, 2, n, T, F]
    out = {}
    for s, name in enumerate(SOURCES):
        m = masks[s].reshape(2, -1, F).transpose(1, 2, 0)[:frames]  # [frames, F, 2]
        m = np.concatenate([m, np.zeros((frames, N // 2 + 1 - F, 2), np.float32)], 1)
        if a.float64:
            ref = istft64(stft * m)[N:N + x.shape[0]]
        else:
            y = tf.signal.inverse_stft(tf.transpose(stft * m, [2, 0, 1]), N, HOP, window_fn=hann) * (2 / 3)
            ref = tf.transpose(y).numpy()[N:N + x.shape[0]]
        got = read(os.path.join(a.stems, f"{a.prefix}-{name}.wav"))
        n = min(len(ref), len(got))
        err = got[:n] - np.clip(ref[:n], -1, 1)
        if a.write_ref:
            write(os.path.join(a.write_ref, f"ref-{name}.wav"), ref)
        seg = T * HOP
        per = [round(snr(ref[i:i + seg], err[i:i + seg]), 1) for i in range(0, n, seg)] if a.segments else None
        out[name] = {"snr_db_per_chunk": per, "frames_ref": len(ref), "frames_got": len(got),
                     "snr_db": snr(ref[:n], err),
                     "max_abs_diff": float(np.abs(err).max()), "rms": float(np.sqrt((ref ** 2).mean()))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
