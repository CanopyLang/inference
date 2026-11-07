// NS17b bench: separate one whole track with the stem host (external/stem), twice in one profile (the
// first run fills OPFS from the chunked mount, the second must read it), and post per-stage timings and
// memory to separate-serve.py, then the last run's stems. Query: track, prefer, power, threads, runs.
import { separate } from "./_/stem-separator.mjs";

const q = new URLSearchParams(location.search);
const log = (s) => { document.getElementById("log").textContent += s + "\n"; };

async function memory() {
  if (!self.crossOriginIsolated || !performance.measureUserAgentSpecificMemory) return null;
  try { return (await performance.measureUserAgentSpecificMemory()).bytes; } catch { return null; }
}

async function main() {
  const runs = Number(q.get("runs") || 2), r = { track: q.get("track"), prefer: q.get("prefer") || "auto",
    power: q.get("power") || "high-performance", ua: navigator.userAgent, isolated: self.crossOriginIsolated,
    cores: navigator.hardwareConcurrency, runs: [] };
  let last = null;
  // Let the loaded page settle, then mark it: the server's PSS sample here is the baseline.
  await new Promise((ok) => setTimeout(ok, 3000));
  await fetch("/mark/baseline", { method: "POST", body: "" });
  await new Promise((ok) => setTimeout(ok, 500));
  try {
    for (let i = 0; i < runs; i++) {
      const res = await separate(`./tracks/${r.track}`, {
        base: "./_/", worker: "stem-worker.mjs", descriptor: q.get("descriptor") || "spleeter4.stem.json",
        ort: { module: "ort/ort.webgpu.min.mjs", wasmPaths: "ort/" }, prefer: r.prefer, power: r.power,
        threads: q.get("threads") ? Number(q.get("threads")) : undefined, outPrefix: `run${i}`,
      });
      const stems = res.stems.map(({ blob, ...s }) => s);
      r.runs.push({ ...res, stems, uaMemoryBytesAfter: await memory() });
      log(JSON.stringify(r.runs.at(-1)));
      last = res;
    }
    r.ok = true;
  } catch (e) {
    r.ok = false; r.error = { code: e.code, message: String(e.message || e), attempts: e.attempts };
  }
  if (last) for (const s of last.stems) await fetch(`/stem/${s.name}`, { method: "POST", body: s.blob });
  await fetch("/result", { method: "POST", body: JSON.stringify(r) });
  log("posted");
}
main();
