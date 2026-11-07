// NS17a spike: time one chunk of a stem model under ORT-web (WebGPU or WASM SIMD).
// Query: model=<file under models/>&ep=webgpu|wasm&threads=<n>&runs=<n>. Posts one JSON
// result to /result (bench-serve.py). Input is seeded noise: speed, plus agreement with ORT CPU
// (check-outputs.py); never separation quality.
import * as ort from "./ort/ort.webgpu.min.mjs";

const q = new URLSearchParams(location.search);
const model = q.get("model"), ep = q.get("ep"), threads = Number(q.get("threads") || 1);
const runs = Number(q.get("runs") || 3), power = q.get("power") || "high-performance";
const log = (s) => { document.getElementById("log").textContent += s + "\n"; };
const shapes = { mix: [1, 2, 343980], mag: [2, 1, 512, 1024] };

async function post(r) {
  await fetch("/result", { method: "POST", body: JSON.stringify(r) });
  log(JSON.stringify(r));
}

async function main() {
  const r = { model, ep, threads, runs, power, ua: navigator.userAgent, isolated: self.crossOriginIsolated,
              cores: navigator.hardwareConcurrency };
  try {
    ort.env.wasm.wasmPaths = new URL("./ort/", location.href).href;
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.simd = true;
    if (ep === "webgpu") {
      if (!navigator.gpu) throw new Error("webgpu_unavailable: navigator.gpu is undefined");
      const ad = await navigator.gpu.requestAdapter({ powerPreference: power });
      if (!ad) throw new Error("webgpu_unavailable: no adapter");
      const info = ad.info || (ad.requestAdapterInfo ? await ad.requestAdapterInfo() : {});
      r.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
      r.adapterMaxBuffer = ad.limits.maxBufferSize;
      r.adapterMaxStorageBinding = ad.limits.maxStorageBufferBindingSize;
      ort.env.webgpu.powerPreference = power;
    }
    let t = performance.now();
    const res = await fetch(`./models/${model}`);
    if (!res.ok) throw new Error(`model_fetch_failed:${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    r.fetchMs = performance.now() - t; r.modelBytes = bytes.byteLength;
    t = performance.now();
    const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
    r.createMs = performance.now() - t;
    const name = s.inputNames[0], dims = shapes[name];
    if (!dims) throw new Error(`input_unknown:${name}`);
    const n = dims.reduce((a, b) => a * b, 1), data = new Float32Array(n);
    // Seeded LCG, mirrored in check-outputs.py, so a browser output can be compared with ORT CPU.
    let seed = 12345;
    for (let i = 0; i < n; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; data[i] = seed / 4294967296 * 2 - 1; }
    if (name === "mag") for (let i = 0; i < n; i++) data[i] = Math.abs(data[i]);
    const feeds = { [name]: new ort.Tensor("float32", data, dims) };
    const times = [];
    for (let i = 0; i <= runs; i++) {
      t = performance.now();
      const out = await s.run(feeds);
      const o = out[s.outputNames[0]];
      await o.getData?.();
      times.push(performance.now() - t);
      if (i === 0) {
        const d = o.data; let nan = 0, sum = 0;
        for (let k = 0; k < d.length; k++) { if (Number.isNaN(d[k])) nan++; else sum += Math.abs(d[k]); }
        r.outDims = o.dims; r.outNaN = nan; r.outAbsMean = sum / Math.max(1, d.length - nan);
        const step = Math.floor(d.length / 64);
        r.outSample = Array.from({ length: 64 }, (_, k) => d[k * step]);
      }
    }
    r.firstRunMs = times[0];
    const rest = times.slice(1).sort((a, b) => a - b);
    r.runMs = rest; r.medianMs = rest[Math.floor(rest.length / 2)];
    r.audioSeconds = name === "mix" ? 343980 / 44100 : 512 * 1024 / 44100;
    r.realtimeFactor = r.audioSeconds * 1000 / r.medianMs;
    r.ok = true;
  } catch (e) {
    r.ok = false; r.error = String(e && e.message || e);
  }
  await post(r);
}
main();
