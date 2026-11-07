// canopy/inference stem host: model bytes from the mount (edsite docs/canopy-mount-contract.md §6),
// one file or a chunk manifest, each chunk checked with WebCrypto, kept in OPFS keyed by the whole
// file's sha256 so a second visit downloads nothing. A mismatch is a named error, never a retry loop.

const DIR = "canopy-models";

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
export const sha256 = async (bytes) => hex(await crypto.subtle.digest("SHA-256", bytes));

async function fetchBytes(url, what) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${what}_fetch_failed:${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function opfsDir() {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(DIR, { create: true });
  } catch {
    return null; // private mode or no OPFS: download every visit, said in the result as cache "none"
  }
}

async function readJson(dir, name) {
  try { return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text()); } catch { return null; }
}

async function writeJson(dir, name, v) {
  const h = await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle();
  const b = new TextEncoder().encode(JSON.stringify(v));
  h.truncate(0); h.write(b, { at: 0 }); h.flush(); h.close();
}

// spec: { sha256, bytes, url } (one file) or { manifest: url } (contract §6 chunk manifest).
// Returns { bytes, sha256, cache: "hit"|"filled"|"none", fetched, chunks }.
export async function loadModel(spec, base, onProgress = () => {}) {
  let m;
  if (spec.manifest) {
    const murl = new URL(spec.manifest, base);
    const raw = await fetchBytes(murl, "manifest");
    m = JSON.parse(new TextDecoder().decode(raw));
    m.base = murl;
    if (spec.manifestSha256 && (await sha256(raw)) !== spec.manifestSha256) throw new Error("manifest_sha_mismatch");
  } else {
    m = { sha256: spec.sha256, bytes: spec.bytes, chunks: [{ sha256: spec.sha256, bytes: spec.bytes, url: spec.url }], base };
  }
  const dir = await opfsDir();
  const out = new Uint8Array(m.bytes);
  const name = `${m.sha256}.bin`, doneName = `${m.sha256}.done.json`;
  let fetched = 0, cache = dir ? "hit" : "none";
  let h = null, done = [];
  if (dir) {
    h = await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle();
    done = (await readJson(dir, doneName)) || [];
    if (h.getSize() !== m.bytes) { h.truncate(m.bytes); done = []; }
  }
  try {
    let at = 0;
    for (let i = 0; i < m.chunks.length; i++) {
      const c = m.chunks[i], view = out.subarray(at, at + c.bytes);
      if (h && done.includes(i)) {
        h.read(view, { at });
      } else {
        if (cache === "hit") cache = "filled";
        const b = await fetchBytes(new URL(c.url || `${c.sha256}.part`, m.base), "model_chunk");
        if (b.byteLength !== c.bytes || (await sha256(b)) !== c.sha256) throw new Error(`model_chunk_sha_mismatch:${i}`);
        view.set(b);
        fetched += b.byteLength;
        if (h) { h.write(b, { at }); h.flush(); done.push(i); await writeJson(dir, doneName, done); }
      }
      at += c.bytes;
      onProgress({ stage: "model", done: at, total: m.bytes });
    }
  } finally {
    if (h) h.close();
  }
  if ((await sha256(out)) !== m.sha256) {
    if (dir) { await dir.removeEntry(name).catch(() => {}); await dir.removeEntry(doneName).catch(() => {}); }
    throw new Error("model_sha_mismatch");
  }
  return { bytes: out, sha256: m.sha256, cache, fetched, chunks: m.chunks.length };
}
