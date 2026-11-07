// canopy/inference stem host: a 16-bit stereo WAV written as it is produced (NS17b), to an OPFS
// sync access handle when the worker has one, else to Blob parts, so a stem never sits whole in
// the JS heap on the OPFS path.

const HEADER = 44;

function header(rate, frames) {
  const b = new DataView(new ArrayBuffer(HEADER)), data = frames * 4;
  const str = (o, s) => { for (let i = 0; i < s.length; i++) b.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); b.setUint32(4, 36 + data, true); str(8, "WAVE");
  str(12, "fmt "); b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 2, true);
  b.setUint32(24, rate, true); b.setUint32(28, rate * 4, true); b.setUint16(32, 4, true); b.setUint16(34, 16, true);
  str(36, "data"); b.setUint32(40, data, true);
  return new Uint8Array(b.buffer);
}

export class WavSink {
  // dir: an OPFS directory handle or null (Blob parts).
  static async open(dir, name, rate) {
    const s = new WavSink();
    Object.assign(s, { name, rate, frames: 0, parts: [], handle: null, file: null });
    if (dir) {
      s.file = await dir.getFileHandle(name, { create: true });
      s.handle = await s.file.createSyncAccessHandle();
      s.handle.truncate(0);
      s.handle.write(header(rate, 0), { at: 0 });
    }
    return s;
  }

  write(left, right) {
    const n = left.length, pcm = new Int16Array(n * 2);
    for (let i = 0; i < n; i++) {
      pcm[2 * i] = Math.max(-32768, Math.min(32767, Math.round(left[i] * 32767)));
      pcm[2 * i + 1] = Math.max(-32768, Math.min(32767, Math.round(right[i] * 32767)));
    }
    if (this.handle) this.handle.write(new Uint8Array(pcm.buffer), { at: HEADER + this.frames * 4 });
    else this.parts.push(pcm);
    this.frames += n;
  }

  // Returns the finished WAV as a File (OPFS) or Blob (in memory).
  async close() {
    const h = header(this.rate, this.frames);
    if (this.handle) {
      this.handle.write(h, { at: 0 });
      this.handle.flush();
      this.handle.close();
      return await this.file.getFile();
    }
    return new Blob([h, ...this.parts], { type: "audio/wav" });
  }
}
