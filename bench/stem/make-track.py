#!/usr/bin/env python3
"""A seeded synthetic stereo track (drums, bass, a voiced lead, chords) as 16-bit 44.1 kHz WAV, for timing
and host-agreement runs of the stem host. With --voice, a real spoken voice (any file ffmpeg reads) replaces
the lead and is also written alone as OUT.voice.wav, the truth a vocals stem is scored against. Neither says
anything about separation quality on music (NS17d's blind comparison does).

usage: make-track.py OUT.wav [--seconds 180] [--voice FILE]
"""
import argparse, subprocess, wave

import numpy as np

SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=180.0); ap.add_argument("--voice")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    n = int(a.seconds * SR)
    t = np.arange(n) / SR
    beat = 60 / 120
    mix = np.zeros((n, 2))
    pos = t % beat
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-pos * 30)) * pos) * np.exp(-pos * 12)
    hat_env = np.exp(-((t + beat / 2) % beat) * 60)
    hat = rng.standard_normal(n) * hat_env * 0.15
    roots = np.array([41.2, 55.0, 49.0, 36.7])
    bar = (t // (4 * beat)).astype(int) % 4
    bass = 0.3 * (2 * ((t * roots[bar]) % 1) - 1) * (0.6 + 0.4 * np.exp(-pos * 4))
    f0 = 220 * 2 ** (np.array([0, 3, 5, 7, 10, 7, 5, 3])[(t // beat).astype(int) % 8] / 12)
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t))) / SR
    lead = sum(np.sin(k * phase) / k ** 1.3 for k in range(1, 12)) * 0.12
    chord = sum(np.sin(2 * np.pi * roots[bar] * 4 * r * t) for r in (1, 1.26, 1.5)) * 0.05
    voice = np.zeros((n, 2))
    if a.voice:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.voice, "-t", str(a.seconds), "-ac", "1", "-ar", str(SR),
                              "-f", "s16le", "-"], check=True, capture_output=True).stdout
        v = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
        v = v[:n] / max(np.abs(v).max(), 1e-9) * 0.35
        voice[:len(v), 0] = voice[:len(v), 1] = v
        lead = lead * 0
    mix[:, 0] = kick * 0.5 + hat * 0.8 + bass + lead * 0.9 + chord * 1.1 + voice[:, 0]
    mix[:, 1] = kick * 0.5 + hat * 1.2 + bass + lead * 1.1 + chord * 0.9 + voice[:, 1]
    g = 1 / (np.abs(mix).max() * 1.12)
    write(a.out, mix * g)
    if a.voice:
        write(a.out[:-4] + ".voice.wav", voice * g)


def write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).round().astype("<i2").tobytes())


if __name__ == "__main__":
    main()
