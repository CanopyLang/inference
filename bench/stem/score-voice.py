#!/usr/bin/env python3
"""Score a vocals stem against the voice make-track.py --voice mixed in: SDR (dB, scale-invariant) of the
vocals stem and of the other three stems summed, against the voice. A working host puts the voice in
vocals (high SDR there, low in the rest). A sanity check of the wiring, not a quality measure.

usage: score-voice.py TRACK.voice.wav STEM_DIR PREFIX
"""
import argparse, json, os, wave

import numpy as np


def read(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2).astype(np.float64) / 32767


def si_sdr(est, ref):
    est, ref = est.ravel(), ref.ravel()
    a = (est @ ref) / max(ref @ ref, 1e-20)
    t = a * ref
    return float(10 * np.log10((t @ t) / max((est - t) @ (est - t), 1e-20)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice"); ap.add_argument("stems"); ap.add_argument("prefix")
    a = ap.parse_args()
    v = read(a.voice)
    st = {s: read(os.path.join(a.stems, f"{a.prefix}-{s}.wav"))[:len(v)] for s in ("vocals", "drums", "bass", "other")}
    rest = st["drums"] + st["bass"] + st["other"]
    e = lambda x: float((x ** 2).sum())
    print(json.dumps({"vocals_si_sdr_db": si_sdr(st["vocals"], v), "rest_si_sdr_db": si_sdr(rest, v),
                      "voice_energy_share_in_vocals": e(st["vocals"]) / max(e(st["vocals"]) + e(rest), 1e-20)}, indent=1))


if __name__ == "__main__":
    main()
