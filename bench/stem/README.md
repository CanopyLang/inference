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
  - Run the host STFT (Hann 4096 / hop 1024, first 1024 bins), then non-overlapping 512-frame chunks (spleeter itself does not overlap segments; the ISTFT's overlap-add runs across chunk edges).
  - Multiply each mask by the complex STFT, then ISTFT with overlap-add.
  - Write four stems to OPFS as they finish, so no stem sits whole in memory (the decoded input does, at 8 bytes per stereo frame).
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

## 7. NS17b: the host as built (2026-11-07)

**What landed.** `src/StemSeparator.can` (`separate : Config -> File -> Task Error Separation`) over
`external/stem-separator.js`, which imports the engine from the mount's `_/`:

- `external/stem/stem-separator.mjs` decodes the file (Web Audio, 44.1 kHz stereo; mono is duplicated) on the
  page and transfers the PCM to `stem-worker.mjs`, a module Worker.
- `model-store.mjs` loads the model from its chunk manifest (contract §6): each `.part` is SHA-256-checked, kept
  in OPFS under its hash, and the whole file is checked again; a second run reads OPFS and fetches nothing.
- `pickBackend` creates the WebGPU session first and runs the descriptor's reference chunk; it is used only if
  its 512 reference values agree within 1e-3, else WASM SIMD (`min(cores, 8)` threads when cross-origin
  isolated). No agreeing backend is the named error `no_backend_agrees`.
- `spleeter-dsp.mjs` is spleeter's own framing (4096 zeros prepended, periodic Hann 4096 / hop 1024,
  `pad_end`, first 1024 bins, masks zero above, ISTFT window x 2/3, crop) on a radix-2 FFT (`fft.mjs`), one
  chunk at a time. `wav.mjs` streams each stem to OPFS as 16-bit WAV.
- `make-mount.py` writes the mount files: 10 chunks of 16 MiB, the manifest and the `spleeter4.stem.json`
  descriptor with the ORT-CPU reference.

**Timings and memory** (`separate.html` + `separate-serve.py`, headless Chrome 153, ORT-web 1.20.1, build
box: 32 cores, NVIDIA Ampere and Intel Gen12LP adapters; 3:00 track, 16 chunks, seconds; run 2 reads the model
from OPFS). "Peak" is the Chrome process tree's PSS peak over its idle baseline, sampled every 100 ms.

| case | run | backend | model load | session+verify | STFT | infer | ISTFT | write | decode | total | peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| voice-180.m4a:auto | 1 | webgpu (nvidia) | 2.8 (filled) | 2.3 | 1.8 | 6.9 | 5.9 | 0.1 | 0.5 | 20.6 | 1.21 GiB |
| voice-180.m4a:auto | 2 | webgpu (nvidia) | 0.3 (hit) | 1.8 | 1.2 | 8.1 | 4.5 | 0.2 | 0.5 | 16.8 |  |
| voice-180.m4a:webgpu::low-power | 1 | webgpu (intel) | 2.7 (filled) | 3.5 | 1.2 | 34.3 | 4.5 | 0.1 | 0.4 | 46.8 | 1.04 GiB |
| voice-180.m4a:webgpu::low-power | 2 | webgpu (intel) | 0.2 (hit) | 3.1 | 1.1 | 34.1 | 4.3 | 0.0 | 0.3 | 43.2 |  |
| voice-180.m4a:wasm:8 | 1 | wasm x8 | 3.0 (filled) | 1.1 | 1.2 | 7.9 | 4.5 | 0.4 | 0.4 | 18.6 | 0.96 GiB |
| voice-180.m4a:wasm:8 | 2 | wasm x8 | 0.2 (hit) | 1.1 | 1.0 | 6.4 | 3.7 | 0.1 | 0.4 | 13.0 |  |
| voice-180.m4a:wasm:4 | 1 | wasm x4 | 3.5 (filled) | 1.4 | 1.0 | 9.9 | 3.6 | 0.1 | 0.9 | 20.8 | 0.88 GiB |
| voice-180.m4a:wasm:4 | 2 | wasm x4 | 0.2 (hit) | 1.2 | 1.1 | 9.8 | 4.1 | 0.1 | 0.3 | 17.0 |  |
| voice-180.m4a:wasm:1 | 1 | wasm x1 | 4.3 (filled) | 3.8 | 1.4 | 27.9 | 4.4 | 0.1 | 0.8 | 42.9 | 0.85 GiB |
| voice-180.m4a:wasm:1 | 2 | wasm x1 | 0.2 (hit) | 2.6 | 1.1 | 25.9 | 4.0 | 0.3 | 0.3 | 34.5 |  |
| synth-180.wav:wasm:8 | 1 | wasm x8 | 1.0 (filled) | 1.1 | 1.1 | 6.8 | 4.0 | 0.1 | 0.3 | 14.6 | 0.96 GiB |
| synth-180.wav:wasm:8 | 2 | wasm x8 | 0.2 (hit) | 1.2 | 1.2 | 6.7 | 4.1 | 0.1 | 0.2 | 13.8 |  |
| synth-180.wav:auto | 1 | webgpu (nvidia) | 2.5 (filled) | 1.5 | 1.1 | 5.5 | 3.9 | 0.1 | 0.4 | 15.3 | 1.22 GiB |
| synth-180.wav:auto | 2 | webgpu (nvidia) | 0.3 (hit) | 1.6 | 1.0 | 8.1 | 3.5 | 0.3 | 0.4 | 15.3 |  |

- A 3-minute track separates in 13–21 s on WASM x8 or the discrete GPU, 43–47 s on the integrated GPU and
  35–43 s single-threaded. Every case wrote four 31.75 MB stems.
- WASM x8 matches the discrete GPU here; the GPU's win is on machines with fewer cores. Inference is half the
  time; the JS STFT/ISTFT is most of the rest.
- Peak memory over baseline is 0.85–0.96 GiB on WASM and 1.04–1.22 GiB on WebGPU. The memory gate rows are set
  from it (below).
- The four-file split of §4 (one U-Net resident at a time) was not built: the single file's peak is under
  1.25 GiB, and the split costs four session creates.

**The memory gate** (2026-11-08). The dashboard holds one gate row per backend, migration
`2205_edsite_tools_stem_memory_gates`: `edsite_tools.stem_peak_webgpu_bytes` = 1,304,731,648 and `edsite_tools.stem_peak_wasm_bytes` = 1,028,426,752: the
run's largest `separationPeakBytes` per backend (synth-180.wav, auto and wasm:8). Both are welded (a new model or measurement is a new
migration). The SiteConfig mapper writes them into a site's `thresholds` as `tools.stem_peak_*_bytes` when the tools
plugin is on, and the tool page hands them to the app (contract §6).
- `external/stem/memory-gate.mjs` reads them; a missing one is `memory_gate_missing:<key>` before the track is decoded.
- Before any model byte is fetched, the worker drops each backend whose floor is above `navigator.deviceMemory`
  (Chrome rounds it down to a power of two, so it never overstates). It names the drop in the attempts:
  `device_memory_below_gate:<backend>`. With none left, the error is `device_memory_below_gate`, naming both
  numbers. A browser that reports no memory (Firefox, Safari) is admitted, and the result says
  `device_memory_unreported`.
- `memory-gate.test.mjs` (`node --test`) holds those refusals. `separate-serve.py --thresholds` records each case's
  `gateBytes` and `underGate`. Every case in the table above is under its backend's row, by construction: the
  rows are the table's maxima.

**Agreement with spleeter's own pipeline** (`ref-separate.py`, TF STFT/ISTFT, masks on ORT CPU, SNR in dB per
stem: vocals / drums / bass / other):

| comparison | vocals | drums | bass | other |
|---|---|---|---|---|
| noise floor: ORT CPU, graph optimisation off vs on | 24.7 | 43.9 | 30.4 | 40.5 |
| host on WebGPU (synth track) | 27.1 | 46.3 | 32.7 | 42.9 |
| host on WASM x8 (synth track) | 18.3 | 39.6 | 24.0 | 34.9 |
| host on WebGPU from the AAC file vs the reference from its WAV source (includes the AAC round trip) | 17.5 | 28.0 | 16.0 | 16.6 |

- The host's DSP alone (numpy check, all-ones masks) reconstructs at 150.9 dB, so the DSP is not where the gap
  is. The model is ill-conditioned on these inputs: 1e-6 of input noise, or turning ORT's graph optimisation
  off, moves single mask values by up to 0.37. The noise floor row is that effect on CPU alone.
- WebGPU sits above that floor. WASM sits about 6 dB below it and is unexplained; its reference-chunk error
  is 1.0e-6 (WebGPU 3.6e-6), so it is accumulated divergence, not a wrong kernel.
- **Wiring check on a real voice** (a spoken episode over the synthetic music, `score-voice.py`): the vocals
  stem scores SI-SDR -2.9 dB against the voice and the other three summed -13.9 dB, on every backend.
  Spleeter's own pipeline on the same mix scores -2.7 / -14.1. The voice lands in vocals; the absolute
  quality is the 2019 model's on speech over synthetic tones, not the host's.

**Rerun** (build box, the §6 venv plus ffmpeg and Chrome):

```bash
python make-mount.py models/spleeter4.onnx models/spleeter4.json www/_   # chunks, manifest, descriptor
python make-track.py www/tracks/synth-180.wav
python make-track.py www/tracks/voice-180.wav --voice EPISODE.m4a
# the gate rows as the tool page would hand them over (dashboard box, its DATABASE_URL)
psql "$DATABASE_URL" -Atc "SELECT json_object_agg(replace(id, 'edsite_', ''), dashboard.resolve_gate(id, NULL, ''))
  FROM dashboard.gate_def WHERE id LIKE 'edsite\_tools.stem\_peak\_%'" > thresholds.json
python separate-serve.py --root www --out results.jsonl --stems stems --runs 2 --thresholds thresholds.json \
  voice-180.m4a:auto voice-180.m4a:webgpu::low-power voice-180.m4a:wasm:8 voice-180.m4a:wasm:1 synth-180.wav:auto
python score-voice.py www/tracks/voice-180.voice.wav stems/voice-180.m4a_auto stem
python ref-separate.py --float64 models/spleeter4.onnx www/tracks/synth-180.wav stems/synth-180.wav_auto stem
```
