// canopy/inference — external FFI shim for the RestoreEngine capability.
//
// IMPORTANT: src/RestoreEngine.can does NOT `foreign import` this file. Every native call it
// makes goes through `Native.Module` (canopy/native's external/native-module.js): the generic
// __canopy_call ABI, module "RestoreEngine", methods process/release/deviceTier. The REAL work
// lives in the host module — and unlike the pure-Java capabilities, this one is a REAL C++
// NativeModule (the ORT sibling of EchoModule):
//   • Android/iOS: canopy::RestoreEngineModule  (host/shared/cpp/RestoreEngineModule.{h,cpp})
//     reads input pixels from canopy::globalBlobRegistry() by handle, runs the bundled
//     ESPCN super-resolution-10.onnx via ONNX Runtime on the CPU EP on a worker thread, and
//     puts the restored RGBA blob back into the SAME registry, returning its int handle.
//
// So this file is a THIN companion, kept for two reasons:
//   1. It documents, in one place, the JSON wire shapes the host module must speak.
//   2. It provides the Node TEST-HARNESS mock so canopy/inference's pure logic can be
//      exercised off-device (mirrors image.js's installMock). On Hermes `module` is undefined
//      and this whole block is skipped — the package behaves as if this file were absent.
//
// THE WIRE CONTRACT (host module "RestoreEngine"):
//   process     args {image:<handle>, options:{upscale,restoreFaces,colorize,strength}}
//                                                   -> {image:<handle>, width, height}
//   release     args {image:<handle>}              -> null
//   deviceTier  args {}                            -> {tier:"cpu"|"nnapi"|"gpu"}
//
// Binary NEVER appears here: only int handles + JSON. A handle names an "rgba8" Blob in the
// native BlobRegistry (canopy/native host/shared/cpp/CanopyBlobs.h), the SAME registry
// canopy/image's decode/resize mint into — so a handle from Image.decode is a valid `image`
// arg here, and the restored handle renders through canopy/image's CanopyBitmap view.
//
// MODEL NOTE: super-resolution-10.onnx is ESPCN — single-channel (Y) 3x super-resolution. It
// is a STAND-IN, not a face-restoration/colorization model: the host honors `upscale`+`strength`
// and accepts `restoreFaces`/`colorize` as no-ops. Input tensor is [1,1,224,224] (Y resized),
// output [1,1,672,672]; the host recombines with bicubic CbCr to a plausible RGBA result.

// ----------------------------------------------------------------------------
// Node test-harness mock. Registers an in-memory "RestoreEngine" module on the same
// mock-native bus the canopy/native harness installs, so RestoreEngine.process/release/
// deviceTier resolve without a device (and without ORT). Pixels are faked as {w,h} records
// keyed by a fake handle; the shapes match the wire contract above so the Canopy decoders are
// exercised for real. To round-trip a handle produced by image.js's mock, share the bus.
// ----------------------------------------------------------------------------

function _RestoreEngine_installMock(bus) {
    // `bus` is harness/mock-native-modules.js's registry: bus.register(name, handler) where
    // handler(method, args, resolve) and resolve(errJson, resultJson) mirrors __canopy_resolve.
    var blobs = Object.create(null);
    var nextHandle = 100000; // distinct range from image.js's mock to surface cross-wiring bugs

    function put(w, h) { var id = nextHandle++; blobs[id] = { w: w, h: h }; return id; }

    bus.register('RestoreEngine', function (method, args, resolve) {
        try {
            switch (method) {
                case 'process': {
                    var opts = (args && args.options) || {};
                    // ESPCN is 3x super-res. The mock reports a 3x-upscaled size when upscale is
                    // on, else passthrough — matching what the host produces from the input dims.
                    // The mock has no source dims, so it assumes a 256x256 source like the harness.
                    var srcW = 256, srcH = 256;
                    var scale = opts.upscale === false ? 1 : 3;
                    var w = srcW * scale, h = srcH * scale;
                    var out = put(w, h);
                    return resolve('', JSON.stringify({ image: out, width: w, height: h }));
                }
                case 'release':
                    if (args) { delete blobs[args.image]; }
                    return resolve('', 'null');
                case 'deviceTier':
                    return resolve('', JSON.stringify({ tier: 'cpu' }));
                default:
                    return resolve(JSON.stringify({ code: 'module_not_found', message: 'RestoreEngine.' + method }), '');
            }
        } catch (e) {
            return resolve(JSON.stringify({ code: 'rejected', message: String(e) }), '');
        }
    });
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { installMock: _RestoreEngine_installMock };
}
