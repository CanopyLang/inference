#!/usr/bin/env python3
"""Serve the spike page cross-origin isolated (COOP/COEP as edsite's mount contract §2) and
drive headless Chrome over each (model, ep, threads) case; one JSON line per case to --out.

usage: bench-serve.py --root DIR --out results.jsonl --chrome /usr/bin/google-chrome CASE...
CASE is model:ep:threads[:power], e.g. htdemucs.onnx:wasm:1 or x.onnx:webgpu:1:low-power. Run it from a box shell, never the laptop.
"""
import argparse, http.server, json, os, queue, subprocess, tempfile, threading

results = queue.Queue()


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".wasm": "application/wasm", ".onnx": "application/octet-stream"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        results.put(json.loads(body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--flags", default="--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist")
    ap.add_argument("cases", nargs="+")
    a = ap.parse_args()
    os.chdir(a.root)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with open(a.out, "a") as out:
        for case in a.cases:
            model, ep, threads, *rest = case.split(":")
            power = rest[0] if rest else "high-performance"
            url = f"http://127.0.0.1:{a.port}/bench.html?model={model}&ep={ep}&threads={threads}&runs={a.runs}&power={power}"
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
                p = subprocess.Popen([a.chrome, "--headless=new", f"--user-data-dir={prof}", "--no-first-run",
                                      *a.flags.split(), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    r = results.get(timeout=a.timeout)
                except queue.Empty:
                    r = {"model": model, "ep": ep, "threads": int(threads), "ok": False, "error": "timeout"}
                finally:
                    p.terminate()
                    try:
                        p.wait(30)
                    except subprocess.TimeoutExpired:
                        p.kill()
            r["flags"] = a.flags
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(json.dumps(r), flush=True)
    srv.shutdown()


if __name__ == "__main__":
    main()
