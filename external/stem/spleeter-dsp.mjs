// canopy/inference stem host: Spleeter's STFT front end and masked ISTFT back end, one model chunk
// at a time (NS17b). It mirrors spleeter's EstimatorSpecBuilder: one frame of zeros prepended,
// tf.signal.stft (periodic Hann, pad_end), masks zero above `bins`, Hann ISTFT x 2/3, then the crop.
// Spleeter's segments do not overlap, so neither do ours. Why: inference/bench/stem/README.md §5.
import { makeFFT } from "./fft.mjs";

export class SpleeterChunker {
  // left/right: Float32Array PCM at the model's rate; p: { frame, hop, bins, chunkFrames, sources }.
  constructor(left, right, p) {
    if (left.length !== right.length) throw new Error("pcm_channels_unequal");
    Object.assign(this, { left, right, N: p.frame, hop: p.hop, F: p.bins, T: p.chunkFrames, S: p.sources });
    this.len = left.length;
    this.padded = this.len + this.N;
    this.numFrames = Math.ceil(this.padded / this.hop);
    this.numChunks = Math.ceil(this.numFrames / this.T);
    this.fft = makeFFT(this.N);
    this.win = new Float64Array(this.N);
    for (let i = 0; i < this.N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.N);
    this.re = new Float64Array(this.N);
    this.im = new Float64Array(this.N);
    const spec = 2 * this.T * this.F;
    this.specRe = new Float32Array(spec);
    this.specIm = new Float32Array(spec);
    this.accLen = this.T * this.hop + this.N;
    this.acc = Array.from({ length: this.S * 2 }, () => new Float64Array(this.accLen));
  }

  sample(ch, p) {
    const o = p - this.N;
    return o >= 0 && o < this.len ? ch[o] : 0;
  }

  // Fills mag ([2, 1, T, F], the model input) for chunk ci and keeps its complex spectrum.
  analyse(ci, mag) {
    const { N, F, T, hop, re, im, win, specRe, specIm } = this;
    mag.fill(0); specRe.fill(0); specIm.fill(0);
    for (let t = 0; t < T; t++) {
      const f = ci * T + t;
      if (f >= this.numFrames) break;
      const p0 = f * hop;
      for (let i = 0; i < N; i++) {
        re[i] = this.sample(this.left, p0 + i) * win[i];
        im[i] = this.sample(this.right, p0 + i) * win[i];
      }
      this.fft(re, im, -1);
      // Two real frames in one complex FFT: L = (Z[k] + conj Z[N-k]) / 2, R = (Z[k] - conj Z[N-k]) / 2i.
      const oL = t * F, oR = (T + t) * F;
      for (let k = 0; k < F; k++) {
        const nk = (N - k) & (N - 1);
        const lr = (re[k] + re[nk]) / 2, li = (im[k] - im[nk]) / 2;
        const rr = (im[k] + im[nk]) / 2, ri = (re[nk] - re[k]) / 2;
        specRe[oL + k] = lr; specIm[oL + k] = li; mag[oL + k] = Math.hypot(lr, li);
        specRe[oR + k] = rr; specIm[oR + k] = ri; mag[oR + k] = Math.hypot(rr, ri);
      }
    }
  }

  // masks: [S, 2, 1, T, F]. Overlap-adds every source, then calls emit(s, left, right) with the samples
  // of the original signal this chunk finalised (views into a scratch buffer, valid during the call).
  synthesise(ci, masks, emit) {
    const { N, F, T, hop, re, im, win, specRe, specIm, acc, S } = this;
    const scale = 2 / 3 / N;
    for (let t = 0; t < T; t++) {
      if (ci * T + t >= this.numFrames) break;
      const oL = t * F, oR = (T + t) * F, off = t * hop;
      for (let s = 0; s < S; s++) {
        const mL = (s * 2 * T + t) * F, mR = ((s * 2 + 1) * T + t) * F;
        re.fill(0); im.fill(0);
        // Z = A + iB with A, B the masked L, R spectra, Hermitian-extended; the IFFT's real part is L, imaginary R.
        for (let k = 0; k < F; k++) {
          const a = masks[mL + k], b = masks[mR + k];
          const ar = a * specRe[oL + k], ai = a * specIm[oL + k];
          const br = b * specRe[oR + k], bi = b * specIm[oR + k];
          re[k] = ar - bi; im[k] = ai + br;
          if (k > 0) { re[N - k] = ar + bi; im[N - k] = br - ai; }
        }
        this.fft(re, im, 1);
        const aL = acc[2 * s], aR = acc[2 * s + 1];
        for (let i = 0; i < N; i++) {
          const w = win[i] * scale;
          aL[off + i] += re[i] * w;
          aR[off + i] += im[i] * w;
        }
      }
    }
    const start = ci * T * hop, last = ci === this.numChunks - 1;
    const end = last ? this.padded : start + T * hop;
    // Original sample o is padded sample o + N; emit [max(start, N), min(end, padded)) mapped back.
    const from = Math.max(start, N), to = Math.min(end, this.padded);
    if (to > from) {
      const n = to - from;
      if (!this.outL || this.outL.length < n) { this.outL = new Float32Array(n); this.outR = new Float32Array(n); }
      for (let s = 0; s < S; s++) {
        const aL = acc[2 * s], aR = acc[2 * s + 1];
        for (let i = 0; i < n; i++) { this.outL[i] = aL[from - start + i]; this.outR[i] = aR[from - start + i]; }
        emit(s, this.outL.subarray(0, n), this.outR.subarray(0, n));
      }
    }
    for (const a of acc) { a.copyWithin(0, T * hop); a.fill(0, this.accLen - T * hop); }
  }
}
