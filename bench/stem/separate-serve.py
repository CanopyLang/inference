#!/usr/bin/env python3
"""Serve the NS17b bench cross-origin isolated (COOP/COEP as edsite's mount contract §2), drive headless
Chrome over each case, and sample the Chrome process tree's PSS (/proc smaps_rollup) every 100 ms for the
peak, less the idle page's PSS the page marks before it starts (separationPeakBytes); one JSON line per case to --out, the last run's stems to --stems/<case>/.

usage: separate-serve.py --root DIR --out results.jsonl --stems DIR CASE...   CASE is track:prefer[:threads[:power]]
Run it from a box shell, never the laptop.
"""
import argparse, http.server, json, os, queue, re, subprocess, tempfile, threading, time

results = queue.Queue()
stem_dir = [None]
marks = queue.Queue()  # (name, time) the page posts to /mark/<name>; the sampler pairs them with PSS


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".part": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        m = re.fullmatch(r"/stem/([a-z]+)", self.path)
        k = re.fullmatch(r"/mark/([a-z0-9-]+)", self.path)
        if k:
            marks.put((k.group(1), time.time()))
        elif m:
            open(os.path.join(stem_dir[0], f"stem-{m.group(1)}.wav"), "wb").write(body)
        else:
            results.put(json.loads(body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


def tree(root):
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                ppid = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(p))
            except (OSError, IndexError, ValueError):
                pass
    out, todo = [], [root]
    while todo:
        p = todo.pop(); out.append(p); todo += kids.get(p, [])
    return out


def pss(pids):
    total = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/smaps_rollup"):
                if line.startswith("Pss:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--stems", required=True)
    ap.add_argument("--chrome", default="/usr/bin/google-chrome"); ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--runs", type=int, default=2); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--flags", default="--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist")
    ap.add_argument("cases", nargs="+")
    a = ap.parse_args()
    root, stems, outp = os.path.abspath(a.root), os.path.abspath(a.stems), os.path.abspath(a.out)
    os.chdir(root)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with open(outp, "a") as out:
        for case in a.cases:
            track, prefer, *rest = case.split(":")
            threads = rest[0] if rest and rest[0] else ""
            power = rest[1] if len(rest) > 1 else "high-performance"
            stem_dir[0] = os.path.join(stems, case.replace(":", "_"))
            os.makedirs(stem_dir[0], exist_ok=True)
            url = (f"http://127.0.0.1:{a.port}/separate.html?track={track}&prefer={prefer}&power={power}&runs={a.runs}"
                   + (f"&threads={threads}" if threads else ""))
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
                p = subprocess.Popen([a.chrome, "--headless=new", f"--user-data-dir={prof}", "--no-first-run",
                                      "--enable-blink-features=ForceEagerMeasureMemory", *a.flags.split(), url],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                peak, t0, last, seen = 0, time.time(), 0, {}
                r = None
                while r is None:
                    try:
                        r = results.get(timeout=0.1)
                    except queue.Empty:
                        last = pss(tree(p.pid))
                        peak = max(peak, last)
                        while not marks.empty():
                            name, _ = marks.get()
                            seen[name] = last  # the PSS sampled just after the page marked it
                        if time.time() - t0 > a.timeout:
                            r = {"ok": False, "error": {"code": "timeout"}}
                p.terminate()
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
            base = seen.get("baseline")
            # The separation's own cost: the tree's peak over its PSS on the loaded, idle page.
            r.update({"case": case, "chromePeakPssBytes": peak, "chromeBaselinePssBytes": base,
                      "separationPeakBytes": peak - base if base else None, "marks": seen, "flags": a.flags})
            out.write(json.dumps(r) + "\n"); out.flush()
            print(json.dumps(r), flush=True)
    srv.shutdown()


if __name__ == "__main__":
    main()
