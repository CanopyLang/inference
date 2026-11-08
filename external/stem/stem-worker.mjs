// canopy/inference stem host: the module Worker that loads the model, picks a backend that agrees with
// the model's reference chunk (WebGPU first, WASM SIMD after), and separates one track chunk by chunk,
// writing each source to OPFS as it is finalised (NS17b). Started from the mount's `_/`, never blob:
// (edsite docs/canopy-mount-contract.md §2). Messages: stem-separator.mjs.
import { SpleeterChunker } from "./spleeter-dsp.mjs";
import { loadModel } from "./model-store.mjs";
import { WavSink } from "./wav.mjs";
import { admit, deviceMemoryBytes } from "./memory-gate.mjs";

const now = () => performance.now();
const post = (m, t) => self.postMessage(m, t || []);
const progress = (p) => post({ type: "progress", ...p });

// The reference input: the spike's seeded LCG (bench/stem/bench.mjs), magnitudes are |x|.
function referenceInput(n, seed) {
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; d[i] = Math.abs((seed / 4294967296) * 2 - 1); }
  return d;
}

async function verify(ort, session, desc) {
  const r = desc.reference, dims = [2, 1, desc.chunkFrames, desc.bins];
  const out = await session.run({ [desc.input]: new ort.Tensor("float32", referenceInput(dims.reduce((a, b) => a * b), r.seed), dims) });
  const d = out[desc.output].data;
  let worst = 0;
  for (let i = 0; i < r.indices.length; i++) {
    const e = Math.abs(d[r.indices[i]] - r.values[i]);
    worst = Number.isNaN(e) ? Infinity : Math.max(worst, e);
  }
  return worst;
}

async function adapterInfo(power) {
  if (!self.navigator.gpu) return { error: "webgpu_unavailable:no_navigator_gpu" };
  const ad = await navigator.gpu.requestAdapter({ powerPreference: power });
  if (!ad) return { error: "webgpu_unavailable:no_adapter" };
  const i = ad.info || {};
  return { vendor: i.vendor, architecture: i.architecture, description: i.description };
}

// Tries each backend the memory gate admitted, in order; keeps the first whose reference error is within
// the descriptor's bound. The backends the gate refused open the attempts list, each named.
async function pickBackend(ort, model, desc, opts, timings, memory) {
  const attempts = [...memory.refused];
  for (const ep of memory.admitted) {
    const a = { ep };
    attempts.push(a);
    try {
      if (ep === "webgpu") {
        ort.env.webgpu.powerPreference = opts.power || "high-performance";
        a.adapter = await adapterInfo(ort.env.webgpu.powerPreference);
        if (a.adapter.error) { a.refused = a.adapter.error; continue; }
      } else {
        a.threads = ort.env.wasm.numThreads;
      }
      let t = now();
      const session = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: "all" });
      a.sessionMs = now() - t;
      t = now();
      a.referenceMaxAbsErr = await verify(ort, session, desc);
      a.verifyMs = now() - t;
      if (a.referenceMaxAbsErr <= desc.reference.maxAbsErr) {
        timings.sessionMs = a.sessionMs; timings.verifyMs = a.verifyMs;
        return { session, ep, attempts };
      }
      a.refused = "reference_mismatch";
      await session.release();
    } catch (e) {
      a.refused = `backend_failed:${String((e && e.message) || e).slice(0, 200)}`;
    }
  }
  throw Object.assign(new Error("no_backend_agrees"), { attempts });
}

const orderOf = (prefer) => prefer === "wasm" ? ["wasm"] : prefer === "webgpu" ? ["webgpu"] : ["webgpu", "wasm"];

async function separate(m) {
  const t0 = now(), timings = {};
  // Before any model byte is fetched: a device below every admitted backend's floor stops here.
  const memory = admit(orderOf(m.prefer), m.memoryGate || {}, deviceMemoryBytes(self.navigator));
  const ort = await import(m.ort.module);
  ort.env.wasm.wasmPaths = m.ort.wasmPaths;
  ort.env.wasm.numThreads = m.threads || (self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 8) : 1);
  const descUrl = new URL(m.descriptor, m.base);
  let t = now();
  const desc = await (await fetch(descUrl)).json();
  const model = await loadModel(desc.model, descUrl, progress);
  timings.modelMs = now() - t;
  const picked = await pickBackend(ort, model.bytes, desc, m, timings, memory);
  const modelInfo = { name: desc.name, sha256: model.sha256, bytes: model.bytes.byteLength, cache: model.cache, fetched: model.fetched, chunks: model.chunks };
  model.bytes = null;

  const ck = new SpleeterChunker(m.left, m.right, { frame: desc.frame, hop: desc.hop, bins: desc.bins, chunkFrames: desc.chunkFrames, sources: desc.sources.length });
  const dims = [2, 1, desc.chunkFrames, desc.bins], mag = new Float32Array(dims.reduce((a, b) => a * b));
  let dir = null;
  try { dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("canopy-stems", { create: true }); } catch { dir = null; }
  const sinks = await Promise.all(desc.sources.map((s) => WavSink.open(dir, `${m.outPrefix}-${s}.wav`, desc.sampleRate)));
  Object.assign(timings, { stftMs: 0, inferMs: 0, istftMs: 0 });
  const perChunk = [];
  for (let ci = 0; ci < ck.numChunks; ci++) {
    t = now(); ck.analyse(ci, mag); const a = now() - t;
    t = now();
    const out = await picked.session.run({ [desc.input]: new ort.Tensor("float32", mag, dims) });
    const masks = out[desc.output].data; const r = now() - t;
    t = now(); ck.synthesise(ci, masks, (s, l, rr) => sinks[s].write(l, rr)); const y = now() - t;
    out[desc.output].dispose?.();
    timings.stftMs += a; timings.inferMs += r; timings.istftMs += y;
    perChunk.push([Math.round(a), Math.round(r), Math.round(y)]);
    progress({ stage: "separate", done: ci + 1, total: ck.numChunks });
  }
  t = now();
  const files = await Promise.all(sinks.map((s) => s.close()));
  timings.writeMs = now() - t;
  await picked.session.release();
  timings.workerMs = now() - t0;
  post({
    type: "done",
    result: {
      model: modelInfo, backend: { ep: picked.ep, threads: picked.ep === "wasm" ? ort.env.wasm.numThreads : null, attempts: picked.attempts },
      memory: { deviceBytes: memory.deviceBytes, reading: memory.reading, gateBytes: m.memoryGate[picked.ep] },
      isolated: self.crossOriginIsolated, storage: dir ? "opfs" : "memory",
      frames: ck.len, seconds: ck.len / desc.sampleRate, chunks: ck.numChunks, timings, perChunk,
      stems: desc.sources.map((s, i) => ({ name: s, file: sinks[i].name, bytes: files[i].size, blob: files[i] })),
    },
  });
}

self.onmessage = async (e) => {
  try {
    if (e.data.type === "separate") await separate(e.data);
    else throw new Error(`message_unknown:${e.data.type}`);
  } catch (err) {
    post({ type: "error", error: { code: String((err && err.message) || err).split(":")[0], message: String((err && err.message) || err), attempts: err && err.attempts } });
  }
};
