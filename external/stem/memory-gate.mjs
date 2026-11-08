// canopy/inference stem host: the device-memory floor per backend (NS17b). The floors are the dashboard's
// gate rows edsite_tools.stem_peak_<backend>_bytes, the peak bench/stem/README.md §7 measured, handed to the
// app as the tool's thresholds (edsite docs/canopy-mount-contract.md §6). A missing row refuses by name.

export const GATE_KEYS = Object.freeze({ webgpu: "tools.stem_peak_webgpu_bytes", wasm: "tools.stem_peak_wasm_bytes" });
const GIB = 2 ** 30;

const named = (code, detail) => Object.assign(new Error(detail ? `${code}:${detail}` : code), { code });

/** { webgpu, wasm } bytes from the tool's thresholds object; a missing or non-positive row refuses. */
export function readGate(thresholds) {
  const gate = {};
  for (const [ep, key] of Object.entries(GATE_KEYS)) {
    const v = thresholds && thresholds[key];
    if (!Number.isInteger(v) || v <= 0) throw named("memory_gate_missing", key);
    gate[ep] = v;
  }
  return gate;
}

/** The device memory the browser reports, in bytes, or null when it reports none (Firefox, Safari).
 *  Chrome rounds navigator.deviceMemory down to a power of two, so the reading never overstates. */
export function deviceMemoryBytes(nav) {
  const g = nav && nav.deviceMemory;
  return typeof g === "number" && g > 0 ? Math.round(g * GIB) : null;
}

/** Splits the backend order into those the device's memory admits and those it is below, each named.
 *  An unreported memory admits every backend and says so (reading device_memory_unreported). */
export function admit(order, gate, deviceBytes) {
  const admitted = [], refused = [];
  for (const ep of order) {
    const need = gate[ep];
    if (!Number.isInteger(need)) throw named("memory_gate_missing", GATE_KEYS[ep] || ep);
    if (deviceBytes !== null && deviceBytes < need) {
      refused.push({ ep, refused: `device_memory_below_gate:${ep}`, deviceBytes, gateBytes: need });
    } else admitted.push(ep);
  }
  const reading = deviceBytes === null ? "device_memory_unreported" : "device_memory_reported";
  if (admitted.length === 0) {
    const detail = refused.map((r) => `${r.ep} needs ${r.gateBytes} has ${r.deviceBytes}`).join(", ");
    throw Object.assign(named("device_memory_below_gate", detail), { attempts: refused });
  }
  return { admitted, refused, reading, deviceBytes };
}
