#!/usr/bin/env python3
"""Render a saved replay (scratch/replays/*.json) to an animated GIF (and optionally an MP4) for docs and the README.

  tools/replay_gif.py scratch/replays/play_Gemgrab1_vs_agents_s5_220323.json
  tools/replay_gif.py REPLAY.json --from 30 --to 75 --speed 1.5     # a 45 s highlight, played at 1.5x
  tools/replay_gif.py REPLAY.json --flat --mp4                       # 2D map instead of 3D; also write an .mp4

The viewer (tools/viewer/viewer.html) does the drawing, so the GIF looks exactly like the viewer: headless Chrome
loads the page from a small local server, a capture script steps it tick by tick at a fixed frame rate (not in real
time, so no frames are dropped), and posts every canvas frame back as a PNG; ffmpeg turns the PNGs into a GIF.
Needs google-chrome (or chromium) and ffmpeg on the PATH; uses the GPU (Vulkan), --software if that fails.
"""
import argparse
import http.server
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading

# headless Chrome renders WebGL in software by default; on the GPU (Vulkan) a battle renders ~8x faster
GPU_GL = ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=vulkan", "--enable-features=Vulkan"]
SOFTWARE_GL = ["--enable-unsafe-swiftshader"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
VIEWER = ROOT / "tools/viewer"

# Injected into the page: wait for the 3D assets, then for each output frame seek, draw and post the canvas.
CAPTURE_JS = """
<script>
(async () => {
  const C = %(cfg)s, post = (p, body) => fetch(p, {method: 'POST', body});
  try {
    if (C.flat) $('view3d').checked = false;
    if (C.follow) $('cam').value = 'follow';
    $('hudon').checked = true; $('trails').checked = false; $('labels').checked = C.labels;
    await G3.loaded;
    for (let i = 0; i < 600 && G3.ok && !G3.ready && !C.flat; i++) await new Promise(r => setTimeout(r, 100));
    resize();
    const t0 = frames[0][0], tEnd = frames[frames.length - 1][0], rate = R.meta.tickRate;
    const a = t0 + C.from * rate, b = C.to == null ? tEnd : Math.min(tEnd, t0 + C.to * rate);
    const step = rate * C.speed / C.fps;
    let n = 0;
    for (let t = a; t <= b; t += step, n++) {
      idx = Math.max(0, frameAtTick(t) - (frames[frameAtTick(t)][0] > t ? 1 : 0)); ptick = t;
      const nx = frames[idx + 1];
      draw(nx ? Math.min(1, (t - frames[idx][0]) / Math.max(1, nx[0] - frames[idx][0])) : 0);
      await post('/frame/' + n, await new Promise(r => cv.toBlob(r, 'image/png')));
    }
    // hold the final frame (the result) for a moment
    idx = frames.length - 1; ptick = frames[idx][0]; draw();
    const last = await new Promise(r => cv.toBlob(r, 'image/png'));
    for (let k = 0; k < C.fps * C.hold; k++, n++) await post('/frame/' + n, last);
    await post('/done', JSON.stringify({frames: n, w: cv.width, h: cv.height, d3: use3d()}));
  } catch (e) { await post('/done', JSON.stringify({error: String(e && e.stack || e)})); }
})();
</script>
"""


def build_page(replay: pathlib.Path, cfg: dict) -> bytes:
    data = replay.read_text()
    json.loads(data)   # fail early on a truncated recording
    page = (VIEWER / "viewer.html").read_text().replace("/*REPLAY_JSON*/", data.replace("</", "<\\/"))
    return page.replace("</body>", CAPTURE_JS % {"cfg": json.dumps(cfg)} + "</body>").encode()


def capture(page: bytes, frames_dir: pathlib.Path, size, chrome: str, flags=()) -> dict:
    result, done = {}, threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body, ctype = page, "text/html"
            elif path.startswith("/gfx/"):   # the viewer loads gfx3d.js and the 3D packs from /gfx/ when served
                f = (VIEWER / path[5:]).resolve()
                if not f.is_relative_to(VIEWER) or not f.is_file():
                    return self.send_error(404)
                body, ctype = f.read_bytes(), "application/javascript" if f.suffix == ".js" else "application/octet-stream"
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.startswith("/frame/"):
                n = int(self.path.rsplit("/", 1)[1])
                (frames_dir / f"f{n:05d}.png").write_bytes(body)
                if n % 100 == 0:
                    print(f"  frame {n}", flush=True)
            elif self.path == "/done":
                result.update(json.loads(body))
                done.set()
            self.send_response(204)
            self.end_headers()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen(
            [chrome, "--headless=new", f"--window-size={size[0]},{size[1]}", "--force-device-scale-factor=1",
             "--hide-scrollbars", "--mute-audio", "--no-first-run", *flags,
             f"--user-data-dir={profile}", f"http://127.0.0.1:{srv.server_address[1]}/"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not done.wait(timeout=3600):
                sys.exit("timed out waiting for the capture")
        finally:
            proc.terminate()
            proc.wait()
            srv.shutdown()
    if "error" in result:
        sys.exit("capture failed in the page:\n" + result["error"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, help="output .gif (default: next to the replay)")
    ap.add_argument("--from", dest="start", type=float, default=0, help="start, seconds into the battle")
    ap.add_argument("--to", type=float, help="end, seconds into the battle (default: the end)")
    ap.add_argument("--speed", type=float, default=1.5, help="playback speed (default 1.5x)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=480, help="GIF width in pixels")
    ap.add_argument("--hold", type=float, default=2, help="seconds to hold the final frame")
    ap.add_argument("--window", default="900x1100", help="browser size the viewer draws at, WxH")
    ap.add_argument("--flat", action="store_true", help="2D map instead of the 3D view")
    ap.add_argument("--follow", action="store_true", help="3D camera follows the own hero")
    ap.add_argument("--labels", action="store_true", help="name labels over the heroes")
    ap.add_argument("--mp4", action="store_true", help="also write an .mp4 (full resolution)")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--software", action="store_true",
                    help="render WebGL on the CPU (SwiftShader, ~8x slower) instead of the GPU via Vulkan")
    a = ap.parse_args()

    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome or not shutil.which("ffmpeg"):
        sys.exit("needs google-chrome (or chromium) and ffmpeg on the PATH")
    out = a.out or a.replay.with_suffix(".gif")
    cfg = {"from": a.start, "to": a.to, "speed": a.speed, "fps": a.fps, "hold": a.hold,
           "flat": a.flat, "follow": a.follow, "labels": a.labels}
    frames_dir = pathlib.Path(tempfile.mkdtemp(prefix="replay_gif_"))
    print(f"capturing {a.replay.name} ({'2D' if a.flat else '3D'}) ...")
    info = capture(build_page(a.replay, cfg), frames_dir, [int(v) for v in a.window.split("x")], chrome,
                   SOFTWARE_GL if a.software else GPU_GL)
    print(f"  {info['frames']} frames at {info['w']}x{info['h']}" + ("" if info["d3"] or a.flat else " (3D unavailable: 2D)"))

    src = ["-framerate", str(a.fps), "-i", str(frames_dir / "f%05d.png")]
    scale = f"scale={a.width}:-2:flags=lanczos"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *src, "-vf",
                    f"{scale},split[x][y];[x]palettegen=max_colors=128:stats_mode=diff[p];"
                    f"[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(out)], check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if a.mp4:
        mp4 = out.with_suffix(".mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *src, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    if a.keep_frames:
        print(f"frames kept in {frames_dir}")
    else:
        shutil.rmtree(frames_dir)


if __name__ == "__main__":
    main()
