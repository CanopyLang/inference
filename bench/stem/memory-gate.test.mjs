// node --test bench/stem/memory-gate.test.mjs (on the build box): the memory gate refuses by name.
import test from "node:test";
import assert from "node:assert/strict";
import { admit, deviceMemoryBytes, readGate, GATE_KEYS } from "../../external/stem/memory-gate.mjs";

const GIB = 2 ** 30;
const thresholds = { [GATE_KEYS.webgpu]: 1304731648, [GATE_KEYS.wasm]: 1028426752 };

test("a missing row refuses by its key, never a default", () => {
  assert.throws(() => readGate({}), { code: "memory_gate_missing", message: "memory_gate_missing:tools.stem_peak_webgpu_bytes" });
  assert.throws(() => readGate({ [GATE_KEYS.webgpu]: 1 }), { message: "memory_gate_missing:tools.stem_peak_wasm_bytes" });
  assert.throws(() => readGate(null), { code: "memory_gate_missing" });
  assert.throws(() => readGate({ ...thresholds, [GATE_KEYS.wasm]: 0 }), { code: "memory_gate_missing" });
});

test("a 0.5 GiB device is below both peaks: refused by name, with both numbers", () => {
  const gate = readGate(thresholds);
  assert.throws(() => admit(["webgpu", "wasm"], gate, 0.5 * GIB), (e) => {
    assert.equal(e.code, "device_memory_below_gate");
    assert.deepEqual(e.attempts.map((a) => a.refused), ["device_memory_below_gate:webgpu", "device_memory_below_gate:wasm"]);
    assert.match(e.message, /webgpu needs 1304731648 has 536870912, wasm needs 1028426752 has 536870912/);
    return true;
  });
});

test("a 1 GiB device keeps WASM only; a 2 GiB device both; WebGPU alone on 1 GiB refuses", () => {
  const gate = readGate(thresholds);
  const one = admit(["webgpu", "wasm"], gate, 1 * GIB);
  assert.deepEqual(one.admitted, ["wasm"]);
  assert.deepEqual(one.refused.map((r) => r.refused), ["device_memory_below_gate:webgpu"]);
  assert.deepEqual(admit(["webgpu", "wasm"], gate, 2 * GIB).admitted, ["webgpu", "wasm"]);
  assert.throws(() => admit(["webgpu"], gate, 1 * GIB), { code: "device_memory_below_gate" });
});

test("a device between the two floors keeps WASM and names the refused WebGPU", () => {
  const gate = { webgpu: 1304731648, wasm: 1028426752 };
  const r = admit(["webgpu", "wasm"], gate, 1200 * 2 ** 20);
  assert.deepEqual(r.admitted, ["wasm"]);
  assert.equal(r.refused[0].refused, "device_memory_below_gate:webgpu");
});

test("an unreported memory admits and says so; Chrome's bucket reads as bytes", () => {
  const gate = readGate(thresholds);
  const r = admit(["webgpu", "wasm"], gate, deviceMemoryBytes({}));
  assert.equal(r.reading, "device_memory_unreported");
  assert.equal(r.deviceBytes, null);
  assert.equal(deviceMemoryBytes({ deviceMemory: 4 }), 4 * GIB);
  assert.equal(deviceMemoryBytes({ deviceMemory: 0.25 }), GIB / 4);
});
