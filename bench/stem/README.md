# Stem separation spike (NS17a, 2026-11-07)

Which 4-stem model the in-browser stem splitter can ship, how fast one chunk runs under
onnxruntime-web (WebGPU, WASM SIMD), and how the model is served under edsite's mount contract
(`edsite/docs/canopy-mount-contract.md`, sections 4 to 7). The input is seeded noise: this measures
speed and agreement with ORT CPU, not separation quality.

## 1. Licence decision

| weights | code | weights licence | commercial | evidence |
|---|---|---|---|---|
| htdemucs v4 (`htdemucs`, `_ft`, `_6s`) | MIT | **unclear**: the maintainer says "the license of the weights is not clear", because MusDB holds research-only tracks. The 6-source model was trained on Meta internal data, so CC-BY-NC is likely, and the maintainer grants no licence. facebookresearch/demucs was archived on 2025-01-01. | **no** | https://github.com/adefossez/demucs/blob/main/README.md |
| Spleeter 4stems (deezer v1.4.0 release) | MIT | MIT: "source code and pre-trained models are available on github and distributed under a MIT license". It was trained on Deezer internal data, with no MusDB. | **yes** | https://joss.theoj.org/papers/10.21105/joss.02154 |
| Open-Unmix `umxl` / `umxhq` | MIT | `umxl` is CC BY-NC-SA. `umxhq` states no weight licence and was trained on MUSDB. | no | sigsep/open-unmix-pytorch |

There are no "MIT Demucs weights". The Demucs architecture is MIT, but every published checkpoint
falls under the weights caveat above. An MIT Demucs therefore needs our own training on licensed
audio, which is a later option (NS17e) and not a conversion.

**Decision: ship Spleeter 4stems.** The cost is quality. The Spleeter paper reports these SDRs
(dB, musdb18 test, with MWF):

| | vocals | bass | drums | other |
|---|---|---|---|---|
| Spleeter 4stems | 6.86 | 5.51 | 6.71 | 4.55 |
| Demucs v2 | 7.05 | 6.70 | 7.08 | 4.47 |

htdemucs reports about 9.0 dB on average (arXiv 2211.08553). The gap to Demucs-based competitors
is the risk for NS17d's go/no-go.

The decision is recorded as two `dashboard.media_licence_registry` model_weights portfolio rows
(`htdemucs` commercial_ok=false, `spleeter-4stems` commercial_ok=true). They were added by the
editorial-dashboard migration `*_stem_model_licences.sql`.

## 2. The models as exported

| file | params | bytes | sha256 | fp32 vs torch, max abs err |
|---|---|---|---|---|
| `spleeter4.onnx` (`export_spleeter.py`) | 39,290,900 | 157,280,686 | `68bb9d6a…f737` | 5.4e-4 |
| `htdemucs.onnx` (`export_htdemucs.py`) | 41,984,456 | 304,413,258 | `83fc1b8d…da12` | 5.0e-4 |

Full hashes are in `spleeter4.json` and `htdemucs.json`.

**Spleeter export:**
- It is one graph: the four U-Nets plus spleeter's ratio mask.
  - Input: magnitude `[2, n, 512, 1024]` (frame 4096, hop 1024, the first 1024 bins).
  - Output: masks `[4, 2, n, 512, 1024]` in the order vocals, drums, bass, other.
- One chunk is 512 frames, 11.89 s.
- The STFT and ISTFT stay in the host, as in spleeter itself.

**htdemucs export:**
- It uses the dhunstack fork @6a96f8c0 (adefossez/demucs PR #10, unmerged), which replaces STFT/ISTFT with convolutions.
- The input is fixed at `[1, 2, 343980]` (7.8 s).
- About 136 MB of its 304 MB is the conv-STFT kernels.

**fp16 is broken for both models.** The conversion is `onnxruntime.transformers.float16` with keep_io_types:
- On ORT CPU the output is NaN.
- On WebGPU the output is all zero.

NS17b must use mixed precision (fp32 for the mask/softmax and normalisation ops) and verify the
result before fp16 is shipped. Until then, ship fp32.

## 3. Timing: one chunk (`bench.html`, headless Chrome 153, ORT-web 1.20.1)

**Setup:**
- Box: editorial.remote, i9-13900K, RTX 3060 and Intel UHD 770.
- Pages were served cross-origin isolated (`bench-serve.py`).
- The box was shared: load average was 15 to 24, and another process held the 3060. Read these numbers as a floor, not a figure for a typical reader.
- Each cell is the median of 5 runs after one warm-up, in ms per chunk, with the real-time factor (audio seconds / compute seconds) in brackets.

| model | backend | run 1 | run 2 | matches ORT CPU |
|---|---|---|---|---|
| spleeter4 fp32 | WebGPU, NVIDIA 3060 | 236 (50x) | 542 (22x) | yes, rel err 1.3e-7 |
| spleeter4 fp32 | WebGPU, Intel UHD 770 | 2156 (5.5x) | 2140 (5.6x) | yes, 1.3e-7 |
| spleeter4 fp32 | WASM SIMD, 1 thread | 1225 (9.7x) | 1937 (6.1x) | yes, 4.3e-8 |
| spleeter4 fp32 | WASM SIMD, 4 threads | 516 (23x) | 903 (13x) | yes |
| spleeter4 fp32 | WASM SIMD, 8 threads | 661 (18x) | 734 (16x) | yes |
| htdemucs fp32 | WebGPU, NVIDIA 3060 | 280 | 109 | **no**, rel err 5.9 |
| htdemucs fp32 | WebGPU, Intel UHD 770 | 20335 | 20308 | **no**, output is zero |
| htdemucs fp32 | WASM SIMD, 1 thread | 13083 (0.6x) | 27953 (0.28x) | yes, 8e-4 |
| htdemucs fp32 | WASM SIMD, 4 threads | 7967 (0.98x) | 9208 (0.85x) | yes |
| htdemucs fp32 | WASM SIMD, 8 threads | 5118 (1.5x) | 5132 (1.5x) | yes |

- Session create takes 0.6 to 1.2 s for spleeter4 and 1.2 to 4 s for htdemucs.
- Raw results: `results-run1.jsonl`, `results-run2.jsonl`, and the comparison with ORT CPU in `check.jsonl`.

**Reading the results:**
- **A 3-minute track with spleeter4 is 16 chunks.** That takes about 4 to 9 s on a discrete GPU and about 10 to 15 s on WASM with 4 to 8 threads, before the host STFT/ISTFT. Both backends are viable, so the WASM SIMD fallback is a real product rather than a degraded one.
- **htdemucs on ORT-web WebGPU gives wrong numbers** on both adapters (WASM agrees with CPU). This is a second reason, after the licence, not to ship it. NS17b's host must verify every backend against a fixed reference chunk before trusting it.
- **Memory per spleeter chunk:**
  - The input is 4 MiB and the masks are 16 MiB, at 512 frames.
  - The bound is the 157 MB of weights held by the session.
  - Peak is therefore about 0.4 to 0.5 GB with WASM.
  - The gate row must be set from NS17b's measured peak, not from this estimate.

## 4. Serving the model: the Cloudflare cache limit and the chunk plan

**The size limit is not the problem:**
- The largest cacheable object is 512 MB on Free/Pro/Business and 5 GB on Enterprise.
- spleeter4 fp32 (157 MB) and htdemucs (304 MB) both fit as one object, so the CDN does not force chunking.

**The default cache rules are the problem.** Cloudflare caches by file extension. The default list includes `bin`, `js`, `css` and `woff2`, but **not** `onnx`, `part`, `wasm` or `json`. Without a Cache Rule, every model byte, the ORT `.wasm` files and the chunk manifest are fetched from the origin on every visit. The origin's `Cache-Control: immutable` (contract §5) does not make an extension eligible.

This finding is for edsite (NS16):
- A zone Cache Rule should mark `/tools/*/_/*` "eligible for cache", respecting origin headers, before any mount ships.
- NS16's Range test (§4) should also assert `cf-cache-status: HIT` on the second request.

**Chunk plan anyway** (contract §6), for resumability and OPFS, not for the CDN:
- **spleeter4 fp32 at 16 MiB chunks:**
  - 157,280,686 bytes is 10 chunks (9 × 16,777,216, plus a last chunk of 6,285,742).
  - The chunk manifest goes at `_/<sha256>.json` and the chunks at `_/<sha256>.part`. Each chunk is checked with WebCrypto, written to OPFS keyed by the whole file's sha256, and the whole file is then checked. A second visit downloads nothing.
- **Once a verified fp16 exists:** about 79 MB, which is 5 chunks.
- **Option for lower peak memory:** export the four U-Nets as four files of about 39 MB each (3 chunks each) and run them one after another. The peak then holds one net instead of four, at the cost of four session creates.
  - NS17b measures both and picks by the gate row.
- **Size rows:** `tools.chunk_max_bytes` = 16 MiB, which sits well under `edge.object_max_bytes`. `tools.file_max_bytes` must sit below 157 MB, or the model ships as one Range-served `.onnx` (§6, last bullet). The chunked path is preferred.
- **The contract's sizing example needs a correction:** it assumes "the ~130 MB stem model, htdemucs.onnx". The shipped model is spleeter4, 157 MB at fp32.

## 5. NS17b design (from this spike)

- **The ORT-web host** lives in `external/` beside the RestoreEngine host:
  - It runs in a module Worker loaded from `_/`, never `blob:` (contract §2 CSP).
  - Set `ort.env.wasm.wasmPaths` to the mount's `_/` and use `ort.webgpu.min.mjs`.
  - Try WebGPU first, with a high-performance adapter. Before using it, run one fixed reference chunk and compare the result with a stored digest.
  - On a mismatch, or when no adapter exists, fall back to WASM SIMD with `min(hardwareConcurrency, 8)` threads. Threads need crossOriginIsolated, per contract §2.
- **The StemSeparator:**
  - Decode the audio to 44.1 kHz stereo.
  - Run the host STFT (Hann 4096 / hop 1024, first 1024 bins), then 512-frame chunks with overlap.
  - Multiply each mask by the complex STFT, then ISTFT with overlap-add.
  - Write four stems to OPFS as they finish, so a track never sits whole in memory.
- **The bench page** extends `bench.mjs` to a real 3-minute track and records per-stage timings and peak memory.

## 6. Rerun

Run on the build box, never the laptop. The venv is Python 3.11 with:
- torch 2.5.1+cpu and the demucs fork above
- onnx 1.17.0, onnxruntime 1.20.1, onnxconverter-common 1.15.0
- tensorflow-cpu 2.15.1, numpy 1.26.4

```bash
python export_spleeter.py models && python export_htdemucs.py models
# www/ = bench.html, bench.mjs, ort/ (onnxruntime-web@1.20.1 dist), models -> ../models
python bench-serve.py --root www --out results.jsonl --runs 5 \
  spleeter4.onnx:webgpu:1 spleeter4.onnx:webgpu:1:low-power spleeter4.onnx:wasm:4 htdemucs.onnx:wasm:8
python check-outputs.py models www/results.jsonl
```
