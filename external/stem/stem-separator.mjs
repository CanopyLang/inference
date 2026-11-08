// canopy/inference stem host, main thread (NS17b): decode a track to the model's rate in stereo, hand
// it to stem-worker.mjs, and resolve with the four stems as OPFS-backed Files plus per-stage timings.
// The Canopy face is src/StemSeparator.can through external/stem-separator.js.
//
// opts: { base, worker, descriptor, ort: { module, wasmPaths }, prefer: "auto"|"webgpu"|"wasm",
//         power, threads, outPrefix, onProgress, thresholds }  — URLs resolve against `base` (the mount's
//         `_/`); `thresholds` is the tool's thresholds object, which must hold the memory gate rows.

import { readGate } from "./memory-gate.mjs";

export async function decode(input, rate) {
  const buf = input instanceof ArrayBuffer ? input
    : typeof input === "string" ? await (await fetch(input)).arrayBuffer()
    : await input.arrayBuffer();
  const ctx = new OfflineAudioContext(2, 1, rate);
  let audio;
  try { audio = await ctx.decodeAudioData(buf); }
  catch (e) { throw Object.assign(new Error(`audio_decode_failed:${e.message || e}`), { code: "audio_decode_failed" }); }
  const left = audio.getChannelData(0).slice();
  const right = audio.numberOfChannels > 1 ? audio.getChannelData(1).slice() : left.slice();
  return { left, right, sourceChannels: audio.numberOfChannels };
}

// ORT-web takes a prefix or { mjs, wasm }; the worker resolves nothing itself, so both become absolute.
const absolute = (w, base) => typeof w === "string" ? new URL(w, base).href
  : Object.fromEntries(Object.entries(w).map(([k, v]) => [k, new URL(v, base).href]));

export async function separate(input, opts) {
  const t0 = performance.now(), base = new URL(opts.base, location.href);
  const memoryGate = readGate(opts.thresholds); // refuses memory_gate_missing:<key> before decoding
  const rate = opts.sampleRate || 44100;
  let t = performance.now();
  const pcm = await decode(input, rate);
  const decodeMs = performance.now() - t;
  const worker = new Worker(new URL(opts.worker, base), { type: "module", name: "canopy-stem" });
  try {
    const result = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") opts.onProgress?.(m);
        else if (m.type === "done") resolve(m.result);
        else if (m.type === "error") reject(Object.assign(new Error(m.error.message), m.error));
      };
      worker.onerror = (e) => reject(Object.assign(new Error(`worker_failed:${e.message || "load"}`), { code: "worker_failed" }));
      worker.postMessage({
        type: "separate", base: base.href, descriptor: opts.descriptor,
        ort: { module: new URL(opts.ort.module, base).href, wasmPaths: absolute(opts.ort.wasmPaths, base) },
        prefer: opts.prefer || "auto", power: opts.power, threads: opts.threads, memoryGate,
        outPrefix: opts.outPrefix || "track", left: pcm.left, right: pcm.right,
      }, [pcm.left.buffer, pcm.right.buffer]);
    });
    result.timings.decodeMs = decodeMs;
    result.timings.totalMs = performance.now() - t0;
    result.sourceChannels = pcm.sourceChannels;
    return result;
  } finally {
    worker.terminate(); // the session and model bytes go with it
  }
}
