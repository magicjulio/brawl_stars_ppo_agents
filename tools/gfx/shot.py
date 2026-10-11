#!/usr/bin/env python3
"""Headless screenshot of a local page or URL, so the viewer's WebGL2 3D mode can be checked without a human.

  python3 tools/gfx/shot.py PAGE OUT.png [--wait 4] [--size 1600x1000] [--js "EXPR"]...

PAGE is an .html path (opened as file://, a trailing #hash / ?query is kept: x.html#t=600) or an http(s)/file/chrome URL.
Sequence: load -> --wait SEC (the page runs in real time, rAF loops included) -> --wait-for COND -> each --js EXPR in
order (a returned Promise is awaited, top-level await works) -> 2 animation frames (+ --after SEC if --js) -> screenshot.

  tools/gfx/shot.py scratch/replays/x.html /tmp/a.png                           # whatever the page shows after 4 s
  tools/gfx/shot.py scratch/replays/x.html /tmp/b.png --js "setIdx(900)"        # jump to frame 900 first
  tools/gfx/shot.py scratch/replays/x.html /tmp/c.png --js "\\$('play').click()" --after 3   # play for 3 s, then shoot
  tools/gfx/shot.py scratch/replays/x.html /tmp/d.png --wait-for "G3.ready" --size 1280x800

Console errors and warnings, uncaught exceptions and failed resource loads are printed to stderr (--console all for
everything, --strict makes any error exit 3). Exit 0 = PNG written, 1 = no screenshot.

Browser: first of google-chrome, google-chrome-stable, chromium, chromium-browser, /snap/bin/chromium, microsoft-edge,
/opt/..., then firefox. Chromium family runs --headless=new with SwiftShader (software WebGL2, no GPU needed) and is
driven over the DevTools protocol on --remote-debugging-pipe (stdlib only, no websocket). Firefox is a fallback that
only does `firefox --headless --screenshot` (shot right after load; --wait/--js/--wait-for/console are not available).
--cli drives Chrome with plain flags (--virtual-time-budget) instead: no --js, no console levels, and rAF animation
does not advance in virtual time. Nothing is installed; every run uses a throw-away profile dir that is deleted after, and
a local page (file://, localhost) runs without network access unless --online (else the browser phones its vendor).
"""
import argparse
import base64
import fcntl
import json
import os
import pathlib
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


BROWSERS = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "/snap/bin/chromium",
            "microsoft-edge", "microsoft-edge-stable", "/opt/google/chrome/chrome", "/opt/chromium/chrome",
            "/opt/microsoft/msedge/msedge", "/snap/bin/firefox", "firefox", "/opt/firefox/firefox"]  # /snap/bin first: /usr/bin/firefox is a stub that edits desktop settings

DEFAULT_WAIT = 4

CHROME_FLAGS = [
    "--headless=new", "--no-first-run", "--no-default-browser-check", "--password-store=basic", "--mute-audio",
    "--disable-extensions", "--disable-sync", "--disable-component-update", "--disable-background-networking",
    "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows", "--hide-scrollbars", "--force-device-scale-factor=1",
    # software WebGL2: runs without a GPU or display
    "--disable-gpu-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    "--allow-file-access-from-files",
    # chrome registers for Google push (GCM) a few seconds after start even in a fresh profile; send it nowhere
    "--gcm-checkin-url=http://127.0.0.1:9/", "--gcm-registration-url=http://127.0.0.1:9/", "--gcm-mcs-endpoint=http://127.0.0.1:9/",
]


def die(msg, code=1):
    print(f"shot: {msg}", file=sys.stderr)
    sys.exit(code)


def to_url(s):
    if re.match(r"^(https?|file|data|about|chrome):", s):          # chrome://gpu is handy to see why WebGL is off
        return s
    p, tail = pathlib.Path(s), ""
    if not p.exists():
        m = re.match(r"^(.*?)([#?].*)$", s)
        if not (m and pathlib.Path(m.group(1)).exists()):
            die(f"page not found: {s}")
        p, tail = pathlib.Path(m.group(1)), m.group(2)
    return p.resolve().as_uri() + tail


def is_local(url):
    u = urlsplit(url)
    return u.scheme in ("file", "about", "data", "chrome") or u.hostname in ("localhost", "127.0.0.1", "::1")


def net_flags(url, online):
    """A fresh chrome profile phones Google (component updates, accounts, ...) within seconds even with
    --disable-background-networking. Local pages need no network, so only localhost resolves unless --online."""
    if online or not is_local(url):
        return []
    return ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1"]


def find_browser(explicit=None):
    """-> (path, 'chromium' | 'firefox'): the explicit/env choice, else the first installed one from BROWSERS."""
    chosen = explicit or os.environ.get("SHOT_BROWSER")
    for name in ([chosen] if chosen else BROWSERS):
        path = shutil.which(name) or (name if os.path.isabs(name) and os.access(name, os.X_OK) else None)
        if path:
            return path, ("firefox" if "firefox" in os.path.basename(path) else "chromium")
    die(f"browser not found: {chosen}" if chosen else "no browser found (looked for: " + ", ".join(BROWSERS) + ")")


def profile_dir(base, browser):
    """Fresh throw-away profile dir. A snap browser cannot see hidden dirs or /tmp, so it gets one under ~/snap/<name>/common."""
    snap = re.fullmatch(r"/snap/bin/(chromium|firefox)", browser)
    cands = [pathlib.Path.home() / "snap" / snap.group(1) / "common"] if snap else [base, os.environ.get("SHOT_PROFILE_BASE")]
    for cand in cands:
        if cand and pathlib.Path(cand).is_dir():
            return tempfile.mkdtemp(prefix="shot-", dir=cand)
    return tempfile.mkdtemp(prefix="shot-")


class CDPError(RuntimeError):
    pass


class Page:
    """Minimal Chrome DevTools Protocol client over --remote-debugging-pipe (NUL-terminated JSON on fds 3/4) that also
    collects the page's console output, exceptions and failed loads while it waits."""

    def __init__(self, wfd, rfd, proc):
        self.w, self.r, self.proc, self.buf, self.n, self.sid = wfd, rfd, proc, b"", 0, None
        self.loaded = False
        self.logs = []                               # (kind, severity, text, where)

    # ---- transport ----
    def _send(self, method, params=None):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params or {}}
        if self.sid and not method.startswith(("Target.", "Browser.")):
            msg["sessionId"] = self.sid
        data = json.dumps(msg).encode() + b"\0"
        while data:
            data = data[os.write(self.w, data):]
        return self.n

    def _recv(self, timeout):
        end = time.monotonic() + timeout
        while b"\0" not in self.buf:
            left = end - time.monotonic()
            if left <= 0 or not select.select([self.r], [], [], left)[0]:
                return None
            chunk = os.read(self.r, 1 << 20)
            if not chunk:
                raise CDPError("the browser closed the DevTools pipe (exit code %s)" % self.proc.poll())
            self.buf += chunk
        raw, _, self.buf = self.buf.partition(b"\0")
        return json.loads(raw)

    def call(self, method, params=None, timeout=60):
        mid = self._send(method, params)
        end = time.monotonic() + timeout
        while True:
            m = self._recv(end - time.monotonic())
            if m is None:
                raise TimeoutError(f"{method} did not answer within {timeout:g}s")
            if m.get("id") == mid:
                if "error" in m:
                    raise CDPError(f"{method}: {m['error'].get('message')}")
                return m.get("result", {})
            self._event(m)

    def pump(self, seconds):
        """Let the page run for `seconds`, handling its events."""
        end = time.monotonic() + seconds
        while end - time.monotonic() > 0:
            m = self._recv(end - time.monotonic())
            if m is not None:
                self._event(m)

    def pump_until(self, cond, timeout):
        end = time.monotonic() + timeout
        while not cond():
            left = end - time.monotonic()
            if left <= 0:
                return False
            m = self._recv(min(left, 0.25))
            if m is not None:
                self._event(m)
        return True

    # ---- events ----
    @staticmethod
    def _arg(a):
        if "value" in a:
            return a["value"] if isinstance(a["value"], str) else json.dumps(a["value"])
        return a.get("unserializableValue") or a.get("description") or a.get("type", "?")

    @staticmethod
    def _where(url, line, col=None):
        if not url:
            return ""
        short = url.rsplit("/", 1)[-1][:60] if url.startswith("file:") else url[:90]
        return f"{short}:{line + 1}" + (f":{col + 1}" if col is not None else "") if line is not None else short

    def _event(self, m):
        ev, p = m.get("method"), m.get("params", {})
        if ev == "Page.loadEventFired":
            self.loaded = True
        elif ev in ("Inspector.targetCrashed", "Target.targetCrashed"):
            raise CDPError("the page crashed (renderer died)")
        elif ev == "Page.javascriptDialogOpening":            # alert()/confirm() would block the page forever
            self._send("Page.handleJavaScriptDialog", {"accept": True})
            self.logs.append(("dialog", "warning", f"auto-dismissed {p.get('type')}: {p.get('message', '')}", ""))
        elif ev == "Runtime.consoleAPICalled":
            t = p.get("type", "log")
            sev = {"error": "error", "assert": "error", "warning": "warning"}.get(t, "info")
            fr = (p.get("stackTrace", {}).get("callFrames") or [{}])[0]
            text = " ".join(self._arg(a) for a in p.get("args", []))
            self.logs.append((f"console.{t}", sev, ("assertion failed: " if t == "assert" else "") + text,
                              self._where(fr.get("url"), fr.get("lineNumber"), fr.get("columnNumber"))))
        elif ev == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            ex = d.get("exception", {})
            desc = (ex.get("description") or str(ex.get("value", ""))).strip().splitlines()
            self.logs.append(("exception", "error", f"{d.get('text', '')} {desc[0] if desc else ''}".strip(),
                              self._where(d.get("url"), d.get("lineNumber"), d.get("columnNumber"))))
        elif ev == "Log.entryAdded":                          # browser-generated: failed loads, GL, deprecations, ...
            e = p["entry"]
            if (e.get("url") or "").endswith("/favicon.ico"):  # chrome asks http pages for one; not the page's fault
                return
            sev = {"error": "error", "warning": "warning"}.get(e.get("level"), "info")
            self.logs.append((f"log.{e.get('source')}", sev, e.get("text", ""), self._where(e.get("url"), e.get("lineNumber"))))

    def evaluate(self, expr, timeout=60):
        """-> (value, error text or None). The value is the completion value (statements are fine), a returned Promise is
        awaited, top-level `await` works too. Wrapped in a block so each expression's let/const are its own."""
        src = "{\n" + expr + "\n}"

        def run(repl):
            return self.call("Runtime.evaluate", {"expression": src, "awaitPromise": True, "returnByValue": True,
                                                  "userGesture": True, "replMode": repl}, timeout)
        r = run(False)
        d = r.get("exceptionDetails")
        if d and "await is only valid" in d.get("exception", {}).get("description", ""):
            r = run(True)       # REPL mode compiles top-level await (nothing ran the first time: it was a parse error)
            d = r.get("exceptionDetails")
        if d:
            ex = d.get("exception", {})
            return None, ((ex.get("description") or str(ex.get("value", "")) or d.get("text") or "exception").strip().splitlines() or ["exception"])[0]
        return r.get("result", {}).get("value"), None


def spawn_chrome(cmd, logfile):
    """Start Chrome with --remote-debugging-pipe: it reads commands on fd 3 and answers on fd 4."""
    c_r, c_w = os.pipe()
    r_r, r_w = os.pipe()
    # park the child's ends above fd 9 so dup2 onto 3/4 in the child can never clobber one of them
    c_r2, r_w2 = (fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 10) for fd in (c_r, r_w))
    os.close(c_r)
    os.close(r_w)

    def child():
        os.dup2(c_r2, 3)
        os.dup2(r_w2, 4)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT, close_fds=False,
                            preexec_fn=child, start_new_session=True, env=headless_env())
    os.close(c_r2)
    os.close(r_w2)
    return proc, c_w, r_r


def headless_env():
    """No display-server variables for the child: whatever goes wrong, no window can open on the user's screen."""
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["MOZ_HEADLESS"] = "1"
    return env


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def log_tail(path, n=12):
    try:
        lines = [l for l in pathlib.Path(path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return ""
    return "\n".join("  | " + l[:220] for l in lines[-n:])


def shoot_chromium(chrome, url, out, a, prof):
    w, h = a.size
    cmd = [chrome, *CHROME_FLAGS, *net_flags(url, a.online), "--remote-debugging-pipe", f"--user-data-dir={prof}",
           f"--window-size={w},{h}"]
    if os.geteuid() == 0:
        cmd.append("--no-sandbox")                   # chrome refuses to run as root otherwise
    cmd += a.chrome_arg + ["about:blank"]
    if a.verbose:
        print("shot: " + " ".join(cmd), file=sys.stderr)
    logpath = os.path.join(prof, "chrome.log")
    try:
        proc, wfd, rfd = spawn_chrome(cmd, open(logpath, "wb"))
    except (OSError, subprocess.SubprocessError) as e:
        die(f"cannot start {chrome}: {e}")
    pg = Page(wfd, rfd, proc)
    jserrors = 0
    try:
        ver = pg.call("Browser.getVersion").get("product", "?")
        tid = next((t["targetId"] for t in pg.call("Target.getTargets")["targetInfos"] if t["type"] == "page"), None) \
            or pg.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        pg.sid = pg.call("Target.attachToTarget", {"targetId": tid, "flatten": True})["sessionId"]
        for dom in ("Page", "Runtime", "Log"):
            pg.call(dom + ".enable")
        pg.call("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False})

        pg.loaded = False
        nav = pg.call("Page.navigate", {"url": url})
        if nav.get("errorText"):
            raise CDPError(f"cannot load {url}: {nav['errorText']}")
        if not pg.pump_until(lambda: pg.loaded, a.timeout):
            print(f"shot: warning: no load event after {a.timeout:g}s, shooting anyway", file=sys.stderr)

        pg.pump(a.wait)
        if a.wait_for:
            last_err = []

            def cond_true():
                v, err = pg.evaluate(f"(() => {{ try {{ return !!({a.wait_for}); }} catch (e) {{ return false; }} }})()")
                last_err[:] = [err] if err else []        # a syntax error shows up here, a ReferenceError just means "not yet"
                return v is True
            if not pg.pump_until(cond_true, a.timeout):
                print(f"shot: warning: --wait-for {a.wait_for!r} still false after {a.timeout:g}s, shooting anyway"
                      + (f" ({last_err[0]})" if last_err else ""), file=sys.stderr)
        for i, expr in enumerate(a.js):
            try:
                val, err = pg.evaluate(expr, a.timeout)
            except TimeoutError as e:
                val, err = None, str(e)
            if err:
                jserrors += 1
                print(f"shot: --js[{i}] failed: {err}", file=sys.stderr)
            elif val is not None:
                print(f"js[{i}] => {json.dumps(val)}")
        # let the page present a frame (and any --after settling) before capturing
        try:
            pg.evaluate("new Promise(r => { const t = setTimeout(r, 1500); requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(t); r(); })); })", 10)
        except TimeoutError:
            print("shot: warning: the page is busy (no animation frame within 10s), shooting anyway", file=sys.stderr)
        if a.js:
            pg.pump(a.after)

        fmt = {".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(out.suffix.lower(), "png")
        params = {"format": fmt, "fromSurface": True, "captureBeyondViewport": False}
        if fmt != "png":
            params["quality"] = 92
        data = base64.b64decode(pg.call("Page.captureScreenshot", params, a.timeout)["data"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        pg.pump(0.2)                                  # trailing console output

        gl = ""
        if a.verbose:
            gl = pg.evaluate("""(() => { try { const g = document.createElement('canvas').getContext('webgl2');
              if (!g) return 'NO WEBGL2 CONTEXT'; const e = g.getExtension('WEBGL_debug_renderer_info');
              return g.getParameter(g.VERSION) + ' / ' + (e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER));
            } catch (e) { return 'probe failed: ' + e; } })()""")[0]
        try:
            pg.call("Browser.close", timeout=5)
        except (CDPError, TimeoutError):
            pass
        return data, ver, pg.logs, jserrors, gl
    except (CDPError, TimeoutError, OSError) as e:
        tail = log_tail(logpath)
        die(f"{e}" + (f"\nbrowser log tail:\n{tail}" if tail else "")
            + ("\nhint: sandbox problem? try --chrome-arg=--no-sandbox" if "sandbox" in tail.lower() else ""))
    finally:
        kill_group(proc)
        for fd in (wfd, rfd):
            try:
                os.close(fd)
            except OSError:
                pass


def shoot_cli(chrome, url, out, a, prof):
    """Plain-flags mode: virtual time, so timers run fast-forward but requestAnimationFrame barely advances."""
    if a.js or a.wait_for:
        die("--js / --wait-for need the DevTools mode (drop --cli)")
    w, h = a.size
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [chrome, *CHROME_FLAGS, *net_flags(url, a.online), f"--user-data-dir={prof}", f"--window-size={w},{h}",
           f"--virtual-time-budget={max(1, int(a.wait * 1000))}", "--enable-logging=stderr", "--v=0",
           f"--screenshot={out.resolve()}"]
    if os.geteuid() == 0:
        cmd.append("--no-sandbox")
    cmd += a.chrome_arg + [url]
    if a.verbose:
        print("shot: " + " ".join(cmd), file=sys.stderr)
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=a.timeout + a.wait + 30,
                           start_new_session=True, env=headless_env())
    except subprocess.TimeoutExpired:
        die("browser timed out")
    logs = []
    for m in re.finditer(r'CONSOLE:\d+\] "(.*)", source: (.*?) \((\d+)\)$', r.stderr, re.M):
        txt = m.group(1)                              # the CLI log has no levels (everything is INFO); "Uncaught ..." is the only tell
        logs.append(("console", "error" if txt.startswith("Uncaught") else "unknown", txt,
                     Page._where(m.group(2), int(m.group(3)) - 1)))
    if not out.exists():
        die("no screenshot written:\n" + log_tail_text(r.stderr))
    return out.read_bytes(), "chrome (--cli)", logs, 0, ""


def log_tail_text(text, n=12):
    return "\n".join("  | " + l[:220] for l in [l for l in text.splitlines() if l.strip()][-n:])


def shoot_firefox(ff, url, out, a, prof):
    """firefox --headless --screenshot: the shot is taken right after load; there is no wait / script hook."""
    ignored = [n for n, v in (("--wait", a.wait != DEFAULT_WAIT), ("--js", a.js), ("--wait-for", a.wait_for)) if v]
    print("shot: firefox fallback: screenshot is taken right after load" +
          (f"; ignoring {' '.join(ignored)}" if ignored else "") + "; no console capture", file=sys.stderr)
    w, h = a.size
    png = os.path.join(prof, "out.png")               # written inside prof: a snap Firefox cannot write to hidden dirs or /tmp
    cmd = [ff, "--headless", "--no-remote", "--profile", os.path.join(prof, "profile"),
           f"--window-size={w},{h}", "--screenshot", png, url]
    os.makedirs(os.path.join(prof, "profile"))
    if a.verbose:
        print("shot: " + " ".join(cmd), file=sys.stderr)
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=a.timeout + 30,
                           start_new_session=True, env=headless_env())
    except subprocess.TimeoutExpired:
        die("firefox timed out")
    if not os.path.exists(png):
        die("firefox wrote no screenshot:\n" + log_tail_text(r.stdout + r.stderr))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, out)
    return out.read_bytes(), "firefox", None, 0, ""


def report_logs(logs, mode, levels=True):
    errs = sum(1 for l in logs if l[1] == "error")
    warns = sum(1 for l in logs if l[1] == "warning")
    shown, seen = [], {}
    for kind, sev, text, where in logs:
        if mode == "none" or (mode == "errors" and sev not in ("error", "warning", "unknown")):
            continue
        key = (kind, text, where)
        if key in seen:
            seen[key][0] += 1
            continue
        seen[key] = [1, kind, text, where]
        shown.append(seen[key])
    if shown:
        print(f"shot: page console: {errs} error(s), {warns} warning(s)"
              + ("" if levels else " (--cli log has no levels: showing every line)"), file=sys.stderr)
        for n, kind, text, where in shown[:60]:
            t = text if len(text) < 500 else text[:500] + "..."
            print(f"  [{kind}] {t}" + (f"  ({where})" if where else "") + (f"  x{n}" if n > 1 else ""), file=sys.stderr)
        if len(shown) > 60:
            print(f"  ... {len(shown) - 60} more distinct messages", file=sys.stderr)
    return errs


def png_size(data):
    return struct.unpack(">II", data[16:24]) if data[:8] == b"\x89PNG\r\n\x1a\n" else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("page", help="html file or http(s)/file URL")
    ap.add_argument("out", type=pathlib.Path, help="output .png (.jpg/.webp also work)")
    ap.add_argument("--wait", type=float, default=DEFAULT_WAIT, metavar="SEC", help=f"seconds to let the page run after load (default {DEFAULT_WAIT})")
    ap.add_argument("--size", default="1600x1000", metavar="WxH", help="viewport size (default 1600x1000)")
    ap.add_argument("--js", action="append", default=[], metavar="EXPR", help="expression to run before the shot (repeatable; top-level await ok)")
    ap.add_argument("--wait-for", metavar="COND", help="JS condition to poll until true (after --wait), e.g. \"G3.ready\"")
    ap.add_argument("--after", type=float, default=0.5, metavar="SEC", help="settle time after the --js expressions (default 0.5)")
    ap.add_argument("--timeout", type=float, default=60, metavar="SEC", help="per-step limit: load, --wait-for, --js, screenshot (default 60)")
    ap.add_argument("--console", choices=("errors", "all", "none"), default="errors", help="which page console output to print (default errors = errors + warnings)")
    ap.add_argument("--strict", action="store_true", help="exit 3 if the page logged errors / threw / a --js failed (the PNG is still written)")
    ap.add_argument("--browser", metavar="PATH", help="browser binary (default: auto-detect; env SHOT_BROWSER)")
    ap.add_argument("--chrome-arg", action="append", default=[], metavar="FLAG", help="extra Chrome flag (repeatable), e.g. --chrome-arg=--use-gl=angle")
    ap.add_argument("--cli", action="store_true", help="Chrome via plain flags + --virtual-time-budget (no --js)")
    ap.add_argument("--profile-base", metavar="DIR", help="where the throw-away profile dir is created (default: job scratch dir or $TMPDIR)")
    ap.add_argument("--online", action="store_true", help="allow network access (default: a local page/localhost URL can only reach localhost)")
    ap.add_argument("--keep-profile", action="store_true", help="keep the throw-away profile dir (chrome.log inside) and print its path")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the browser command line and the WebGL2 renderer")
    a = ap.parse_args()

    m = re.fullmatch(r"(\d+)\s*[xX,]\s*(\d+)", a.size)
    if not m or not (64 <= int(m.group(1)) <= 8192 and 64 <= int(m.group(2)) <= 8192):
        die(f"--size must look like 1600x1000 (64..8192), got {a.size!r}")
    a.size = (int(m.group(1)), int(m.group(2)))
    url = to_url(a.page)
    browser, kind = find_browser(a.browser)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # still run the cleanup below

    t0 = time.monotonic()
    prof = profile_dir(a.profile_base, browser)
    try:
        if kind == "firefox":
            data, ver, logs, jserr, gl = shoot_firefox(browser, url, a.out, a, prof)
        elif a.cli:
            data, ver, logs, jserr, gl = shoot_cli(browser, url, a.out, a, prof)
        else:
            data, ver, logs, jserr, gl = shoot_chromium(browser, url, a.out, a, prof)
    finally:
        if a.keep_profile:
            print(f"shot: profile kept: {prof}", file=sys.stderr)
        else:
            shutil.rmtree(prof, ignore_errors=True)

    errs = report_logs(logs, a.console, levels=not a.cli) if logs is not None else None     # None: browser has no console capture
    sz = png_size(data)
    print(f"{a.out}: {'x'.join(map(str, sz)) if sz else a.out.suffix} {len(data) // 1024} KB "
          f"[{ver}, {time.monotonic() - t0:.1f}s, " + (f"{errs} console error(s)" if errs is not None else "console n/a") + "]")
    if sz and sz != a.size:
        print(f"shot: warning: screenshot is {sz[0]}x{sz[1]}, asked for {a.size[0]}x{a.size[1]}", file=sys.stderr)
    if gl:
        print(f"shot: webgl2: {gl}", file=sys.stderr)
    if a.strict and ((errs or 0) or jserr):
        sys.exit(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        die("interrupted", 130)
