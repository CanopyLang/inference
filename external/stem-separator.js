// canopy/inference — FFI for StemSeparator (NS17b). The engine is ES modules served from the mount's
// `_/` (external/stem/*.mjs): this file only loads stem-separator.mjs with import() and turns its
// promise into a Task. Results and errors cross as JSON values the Canopy side decodes.

function _Stem_value(v) { return typeof _Json_unwrap === 'function' ? _Json_unwrap(v) : v; }
function _Stem_wrap(v) { return typeof _Json_wrap === 'function' ? _Json_wrap(v) : v; }

/**
 * @canopy-type Json.Encode.Value -> File -> Task Json.Decode.Value Json.Decode.Value
 * @name separate
 */
var separate = F2(function(config, file) {
    return _Scheduler_binding(function(callback) {
        var c = _Stem_value(config);
        var base;
        try { base = new URL(c.base, location.href); }
        catch (e) { callback(_Scheduler_fail(_Stem_wrap({ code: 'config_invalid', message: String(e) }))); return; }
        import(new URL(c.engine, base).href).then(function(engine) {
            return engine.separate(file, {
                base: base.href, worker: c.worker, descriptor: c.descriptor,
                ort: { module: c.ortModule, wasmPaths: c.ortWasmPaths },
                prefer: c.backend, outPrefix: c.outPrefix, thresholds: c.thresholds
            });
        }).then(function(r) {
            r.stems = r.stems.map(function(s) {
                return { name: s.name, file: s.file, bytes: s.bytes, url: URL.createObjectURL(s.blob) };
            });
            callback(_Scheduler_succeed(_Stem_wrap(r)));
        }, function(e) {
            callback(_Scheduler_fail(_Stem_wrap({
                code: (e && e.code) || String((e && e.message) || e).split(':')[0],
                message: String((e && e.message) || e)
            })));
        });
    });
});

/**
 * @canopy-type String -> Task Never ()
 * @name revoke
 */
function revoke(url) {
    return _Scheduler_binding(function(callback) {
        URL.revokeObjectURL(url);
        callback(_Scheduler_succeed(_Utils_Tuple0));
    });
}
