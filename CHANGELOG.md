# Changelog — canopy/inference

## Unreleased

- The package's `native/cpp` is now the only copy of `RestoreEngineModule.{cpp,h}`,
  `RestoreTiling.h` and `RestoreColor.h` (canopy/native's host copies are deleted). `native.json`
  declares the factory `globalRestoreEngineModule` (header `RestoreEngineModule.h`), so the
  module autolinks with no host wiring; it is one process-wide instance, so model bytes loaded at
  boot survive a runtime reload. `RestoreEngineModule::loadModel` implements the host's generic
  `NativeModule::loadModel`, which `loadModelForModule("RestoreEngine", bytes)` (an app's
  native.json `boot` entry) now reaches. Needs a canopy/native host with that hook.
- The pure tiling and colour tests moved here: `tests/cpp/run.sh` (tilecover, colorconv).

## 2.0.0

- `RestoreEngine.process : Blob.Image -> Options -> (Result NM.Error Blob.Image -> msg) -> Cmd msg`
  takes and returns `Native.Blob.Image` (canopy/native 0.2.0), the shared image handle
  (CONSOLIDATION W2-6).
- Removed `RestoredImage`, `handleToInt`, `width`, `height`, `processBlob`, `release` and
  `bitmap`/`bitmapHandle` (now `Native.Blob.*`), and the native `release` method (C++ and iOS).
- Dropped the canopy/image and canopy/virtual-dom dependencies; requires canopy/native 0.2.0+.

**Migration:** `RestoreEngine.RestoredImage` is now `Native.Blob.Image`. `processBlob` is now `process`. Replace `X.handleToInt h`
with `Native.Blob.toInt h`, `X.release h toMsg` with `Task.attempt toMsg (Native.Blob.release h)`,
`X.bitmap` / `X.bitmapHandle` with `Native.Blob.bitmap` / `Native.Blob.bitmapHandle`, and
`Image.adopt` with `Native.Blob.adopt`.
