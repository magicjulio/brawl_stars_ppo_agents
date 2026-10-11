#!/usr/bin/env python3
"""Play an offline battle yourself in the browser, against the game's bots or against your own trained agents. The
battle runs headless on the tablet (driver `serve`), this script relays your input at 20 ticks/s and the viewer page
draws the result.

  python3 tools/play.py                              # Gem Grab as Shelly, seed 1, bots on both sides
  python3 tools/play.py loc=Ball1 hero=RocketGirl seed=7 diff=2
  python3 tools/play.py --vs                         # you + 2 agents vs 3 agents (newest runs/*/last.pt)
  python3 tools/play.py --vs runs/team45/last.pt --mates bots     # you + 2 bots vs 3 agents
  python3 tools/play.py --mates runs/team45/last.pt  # you + 2 agents vs 3 bots
  python3 tools/play.py --port 8765 --no-open

Agents are PPO checkpoints of tools/train_ppo.py. They decide every 4 ticks from the same observation as in training
(the enemy team sees the battle turned around, so its spawn is at the bottom as it was in training) and need torch:
without it in this python, the script restarts itself with the shared venv ../../.venv/bin/python.

Controls: WASD / arrows move · Space / Enter / J attack with auto-aim · left click attack towards the cursor ·
          E / K super with auto-aim · right click super towards the cursor · Q gadget · P pause · R new battle
Each finished (or restarted) battle is saved to scratch/replays/play_<loc>_s<seed>_<time>.json + .html.
"""
import argparse
import base64
import collections
import hashlib
import json
import os
import pathlib
import queue
import signal
import struct
import subprocess
import sys
import threading
import math
import time
import traceback
import types
import urllib.parse
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsenv import Action, BrawlEnv  # noqa: E402
from bsvec import parse_layout  # noqa: E402
from replay import OUT, VIEWER, build_page  # noqa: E402
from viewer_assets import send_gfx  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV_PY = ROOT.parent.parent / ".venv" / "bin" / "python"
DECIDE_TICKS = 4    # agents decide every 4 ticks, as in training (train_ppo Config.repeat)


class Policy:
    """A tools/train_ppo.py checkpoint playing some brawlers of the battle, sampling its choices as in training/eval."""

    def __init__(self, path, layout):
        import torch
        import train_ppo
        torch.set_num_threads(1)                                  # a few rows per decision; leave the cores alone
        ck = torch.load(path, map_location="cpu")
        self.aim = ck["config"].get("aim", "free")                # runs before the aim setting used the free menu
        blocks, dim, _, _ = parse_layout(layout)                  # a Net, or a TeamNet (one net per brawler)
        self.net = train_ppo.build_net(ck, types.SimpleNamespace(blocks=blocks, obs_dim=dim))
        self.net.eval()
        self.torch, self.tp, self.path = torch, train_ppo, str(path)

    def act(self, vec, grid):
        """vec [rows, obs_dim], grid [rows, G, G] -> one Action per row (in the row's own frame; the driver turns
        the enemy team's back)"""
        with self.torch.inference_mode():
            logits, _ = self.net(self.torch.from_numpy(vec), self.torch.from_numpy(grid))
            choice = self.tp.sample(logits)[0].numpy()
        aim = self.tp.AIM_MENUS[self.aim]
        out = []
        for c in choice:
            mv, atk, sup = self.tp.MOVE[c[0]], aim[c[1]], aim[c[2]]
            out.append(Action(move=(mv[0], mv[1]), attack=int(atk[0]), aim=(atk[1], atk[2]), super_=int(sup[0]),
                              super_aim=(sup[1], sup[2]), gadget=bool(self.tp.GADGET[c[3]] > 0.5)))
        return out


class Session:
    def __init__(self, env, battle, mates=None, foes=None):
        """mates / foes: checkpoint paths for your two teammates / the enemy team, None = the game's bots"""
        self.env, self.battle, self.sent, self.saved = env, battle, 0, True
        self.seed = int(battle.get("seed", 1))
        self.models = {"mates": mates, "foes": foes}
        self.policies = {}                                        # path -> Policy (built once the layout is known)
        self.ticks, self.pending, self.groups = 0, None, {"mates": None, "foes": None}   # groups: rows per side
        self.due = False                                          # the agents decide after this tick's frame is sent

    def _names(self):
        new = [[i, n] for i, n in enumerate(self.env.names) if i >= self.sent]
        self.sent = len(self.env.names)
        return new

    def _policy(self, path):
        if path not in self.policies:
            self.policies[path] = Policy(path, self.env.meta["obs_layout"])
        return self.policies[path]

    def _decide(self):
        """agents look at the battle and choose their next action (sent with the next step)"""
        vec, grid = self.env.observe()
        acts = [None] * len(vec)
        for side, rows in self.groups.items():
            if rows:
                acts[rows.start:rows.stop] = self._policy(self.models[side]).act(vec[rows], grid[rows])
        self.pending = acts

    def think(self):
        """after a step, once its frame is out: the agents decide if it is their turn (so the frame never waits)"""
        if self.due:
            self.due = False
            self._decide()

    def save(self):
        if self.saved or not self.env.frames:
            return None
        OUT.mkdir(parents=True, exist_ok=True)
        vs = "_vs_agents" if self.models["foes"] else ""
        path = OUT / f"play_{self.env.meta['loc']}{vs}_s{self.env.meta['seed']}_{time.strftime('%H%M%S')}.json"
        self.env.save_replay(path)
        build_page(path)
        self.saved = True
        print(f"replay saved: {path.with_suffix('.html')}")
        return str(path.with_suffix(".html"))

    def reset(self, next_seed):
        self.save()
        if next_seed:
            self.seed += 1
        keys = {**self.battle, "seed": self.seed, "agents": 3 if self.models["mates"] else 1,
                "enemies": 3 if self.models["foes"] else 0}
        obs = self.env.reset(**keys)
        self.sent, self.saved, self.ticks, self.pending, self.due = 0, False, 0, None, False
        meta = self.env.meta
        k = keys["agents"]
        self.groups = {"mates": slice(1, k) if k > 1 else None,
                       "foes": slice(k, len(meta["rows"])) if keys["enemies"] else None}
        # who plays whom, for the viewer and the saved replay: "you" / "agent" / "bot" per player index
        meta["control"] = ["bot"] * meta["players"]
        for i, p in enumerate(meta["rows"]):
            meta["control"][p] = "you" if i == 0 else "agent"
        meta["models"] = {side: str(path) for side, path in self.models.items() if path}
        if self.groups["mates"] or self.groups["foes"]:
            self._decide()
        return {"meta": meta, "names": self._names(), "frame": obs.raw, "done": obs.done}

    def step(self, a):
        you = Action(move=(a.get("mx", 0), a.get("my", 0)), attack=a.get("atk", 0), aim=(a.get("ax", 0), a.get("ay", 0)),
                     super_=a.get("sup", 0), super_aim=(a.get("sx", 0), a.get("sy", 0)), gadget=a.get("gadget", 0))
        rows = [you] + (self.pending[1:] if self.pending else [None] * (len(self.env.meta["rows"]) - 1))
        self.pending = None
        obs = self.env.step(rows, repeat=1)
        self.ticks += 1
        self.due = not obs.done and self.ticks % DECIDE_TICKS == 0 and bool(self.groups["mates"] or self.groups["foes"])
        msg = {"names": self._names(), "frame": obs.raw, "done": obs.done}
        if obs.done:
            msg["saved"] = self.save()
        return msg


PRESS_FIELDS = ("atk", "ax", "ay", "sup", "sx", "sy", "gadget")


def num(v):
    """a number from the page: None, NaN, inf or junk -> 0"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


class Game:
    """Runs the battle on its own clock, 20 ticks/s like the game's server, so a slow or jittery network (a phone on
    Wi-Fi) only delays the picture, never the game. Pages send their input with POST /input, watch every tick over the
    server-sent event stream GET /events and reset / pause with POST /control. Only this thread touches the battle."""

    TICK = 0.05

    def __init__(self, session):
        self.session = session
        self.cv = threading.Condition()
        self.move, self.hold, self.presses = (0.0, 0.0), set(), collections.deque()
        self.paused = self.done = self.stopped = False
        self.want_reset = None                    # None, or True = next seed / False = same seed
        self.controller = None                    # page id of the last page that sent input (it may auto-pause)
        self.clients = []                         # one queue of encoded messages per open /events stream
        self.start = None                         # what a stream that joins now gets first: meta, all names, last frame
        self.battle, self.grid, self.last_input = 0, None, 0.0
        self.ready, self.reset_at = 0, 0.0           # newest battle a page has loaded its 3D models for (hold)

    # -- called from the HTTP threads --
    def input(self, body):
        """{"id", "mx", "my", "hold": ["atk", "sup"] (auto-aim every tick while held),
        "press": [{"atk": 1|2, "ax", "ay"} | {"sup": 1|2, "sx", "sy"} | {"gadget": 1}, ...] (each applied once)}"""
        with self.cv:
            self.controller, self.last_input = body.get("id"), time.monotonic()
            self.move = (num(body.get("mx")), num(body.get("my")))
            self.hold = {h for h in body.get("hold", ()) if h in ("atk", "sup")}
            for p in body.get("press", ()):
                if isinstance(p, dict):   # only known fields, as clean numbers (a page may send null / NaN)
                    self.presses.append({k: num(v) for k, v in p.items() if k in PRESS_FIELDS})

    def control(self, body):
        """{"start": true} (a page opened: start the first battle, else join the running one), {"reset": "same" |
        "next"} or {"pause": bool, "id": page id, "auto": bool (page hidden/shown: only the controlling page's counts)}
        or {"ready": battle number} (the page has that battle's 3D models: a new battle waits for it, see holding)"""
        with self.cv:
            if "ready" in body:
                self.ready = max(self.ready, int(num(body["ready"])))
            if body.get("start") and not self.start:
                self.want_reset = False
            if "reset" in body:
                self.want_reset = body["reset"] == "next"
            if "pause" in body and (not body.get("auto") or body.get("id") == self.controller):
                self.paused = bool(body["pause"])
                self._send({"type": "state", "paused": self.paused})
            self.cv.notify_all()

    def subscribe(self):
        q = queue.Queue()
        with self.cv:
            self.clients.append(q)
            if self.start:
                q.put(json.dumps({**self.start, "paused": self.paused}, separators=(",", ":")))
        return q

    def unsubscribe(self, q):
        with self.cv:
            if q in self.clients:
                self.clients.remove(q)

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    # -- the game thread --
    def _send(self, msg):                         # with self.cv held
        data = json.dumps(msg, separators=(",", ":"))
        for q in self.clients:
            q.put(data)

    def _action(self):
        """this tick's input: the held stick and buttons plus at most one waiting press per button"""
        if time.monotonic() - self.last_input > 1.0:   # the page went quiet (no heartbeat): let go of the stick
            self.move, self.hold = (0.0, 0.0), set()
        a = {"mx": self.move[0], "my": self.move[1]}
        for h in self.hold:
            a[h] = 1
        used = set()
        while self.presses:
            p = self.presses[0]
            kind = next((k for k in ("atk", "sup", "gadget") if k in p), None)
            if kind in used:
                break                             # a second press of the same button waits for the next tick
            self.presses.popleft()
            if kind:
                used.add(kind)
                a.update(p)
        return a

    HOLD = 30.0                                   # longest wait for a page to load a new battle's models (s)

    def holding(self):
        """a new battle starts once a page has its 3D models (a phone takes a few seconds), at most HOLD seconds"""
        return self.ready < self.battle and time.monotonic() - self.reset_at < self.HOLD

    def run(self):
        clock = time.perf_counter()
        while True:
            with self.cv:
                while not self.stopped and self.want_reset is None and (self.paused or self.done or not self.start
                                                                         or not self.clients or self.holding()):
                    self.cv.wait(1.0 if self.holding() else None)
                if self.stopped:
                    return
                reset, self.want_reset = self.want_reset, None
                action = self._action() if reset is None else None
            try:
                self._tick(reset, action)
            except Exception as e:                # keep the battle loop alive; show the error in the terminal + page
                traceback.print_exc()
                with self.cv:
                    self._send({"type": "error", "msg": f"{type(e).__name__}: {e}"})
                time.sleep(0.5)
            if reset is None:
                clock += self.TICK
                delay = clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.25:               # fell behind (tablet hiccup): carry on from now, don't race
                    clock = time.perf_counter()
            else:
                clock = time.perf_counter()

    def _tick(self, reset, action):
        """one reset or one game tick, then tell every page"""
        if reset is not None:
            msg = self.session.reset(reset)
            with self.cv:
                self.presses.clear()
                self.done, self.battle, self.grid = msg["done"], self.battle + 1, msg["frame"][4]
                self.reset_at = time.monotonic()
                self.start = {"type": "reset", "battle": self.battle, **msg}
                self._send(self.start)
            return
        msg = self.session.step(action)
        with self.cv:
            self.done = msg["done"]
            frame = msg["frame"]
            if frame[4]:
                self.grid = frame[4]
            else:                                 # a page joining later needs the tile grid in its first frame
                frame = frame[:4] + [self.grid] + frame[5:]
            self.start = {**self.start, "names": self.start["names"] + msg["names"], "frame": frame,
                          "done": msg["done"]}
            self._send({"type": "frame", **msg})
        self.session.think()


MANIFEST = json.dumps({          # "Add to Home Screen" on a phone: starts full screen, without the browser's bars
    "name": "Brawl Stars vs agents", "short_name": "Brawl", "start_url": "/", "display": "fullscreen",
    "orientation": "landscape", "background_color": "#14171c", "theme_color": "#14171c",
    "icons": [{"src": "/icon.png", "sizes": "180x180", "type": "image/png"}]})


def icon_png(n=180):
    """the home screen icon: a purple gem on the viewer's dark background, as a PNG written by hand"""
    rows = b""
    for y in range(n):
        rows += b"\0"
        for x in range(n):
            u, v = abs(x - n / 2) / (n * .3), (y - n * .42) / (n * .42)
            inside = u + abs(v) * (1.6 if v < 0 else 1) < 1 if -0.62 < v < 1 else False
            rows += bytes((192, 132, 252) if inside and x < n / 2 else (150, 90, 230) if inside else (20, 23, 28))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):   # a phone dropped a connection (screen off, app switch)
            log(f"{client_address[0]} connection dropped")
            return
        super().handle_error(request, client_address)


LOG = ROOT / "scratch" / "play_log.txt"      # requests + what the pages report (/log): for debugging phones


def log(line):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + line + "\n")


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"   # RFC 6455


def ws_read(f):
    """one WebSocket message from the page -> (opcode, payload); pages send small unfragmented, masked frames"""
    h = f.read(2)
    if len(h) < 2:
        raise ConnectionError("closed")
    op, n = h[0] & 0x0F, h[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", f.read(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", f.read(8))[0]
    if n > 1 << 20:
        raise ConnectionError("message too large")
    mask = f.read(4) if h[1] & 0x80 else b"\0\0\0\0"
    data = f.read(n)
    if len(data) < n:
        raise ConnectionError("closed")
    return op, bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def ws_frame(data, op=1):
    """a server -> page WebSocket frame (unmasked), as one write"""
    n = len(data)
    head = bytes([0x80 | op, n]) if n < 126 else bytes([0x80 | op, 126]) + struct.pack(">H", n)
    return head + data


def make_handler(game):
    page = VIEWER.read_text().replace("/*REPLAY_JSON*/", json.dumps({"live": True}))
    icon = icon_png()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"             # keep-alive: a phone sends input many times a second
        disable_nagle_algorithm = True            # TCP_NODELAY: small replies and frames leave at once (Nagle +
                                                  # the phone's delayed ACKs held each one back on Wi-Fi)

        def log_message(self, fmt, *args):        # every request except the frequent input, to scratch/play_log.txt
            if "/input" not in self.path or not str(args[1:2]).startswith("('200"):
                log(f"{self.client_address[0]} {fmt % args}")

        def _reply(self, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._reply(page, "text/html; charset=utf-8")
            elif path == "/manifest.webmanifest":
                self._reply(MANIFEST, "application/manifest+json")
            elif path == "/icon.png":
                self._reply(icon, "image/png")
            elif send_gfx(self, path):
                pass
            elif path == "/events":
                self._events()
            elif path == "/ws":
                self._ws()
            else:
                self.send_error(404)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            q = game.subscribe()
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")      # keeps idle connections open
                        self.wfile.flush()
                        continue
                    self.wfile.write(b"data: " + data.encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:                       # the page went away
                pass
            finally:
                game.unsubscribe(q)

        def _ws(self):
            """the page's input as a WebSocket: one ordered stream, so a stick change never waits for the reply to
            the previous one (POST /input does, one request at a time). {"ping": t} is answered with {"pong": t}."""
            key = self.headers.get("Sec-WebSocket-Key")
            if not key or self.headers.get("Upgrade", "").lower() != "websocket":
                self.send_error(400)
                return
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept",
                             base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode())
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    op, data = ws_read(self.rfile)
                    if op == 8:                   # close
                        self.wfile.write(ws_frame(b"", 8))
                        return
                    if op == 9:                   # ping -> pong
                        self.wfile.write(ws_frame(data, 10))
                    elif op == 1:
                        body = json.loads(data)
                        if "ping" in body:
                            self.wfile.write(ws_frame(json.dumps({"pong": body["ping"]}).encode()))
                        else:
                            game.input(body)
            except (OSError, ValueError):         # the page went away, or sent junk
                pass

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if path == "/input":
                game.input(body)
                self._reply("{}")
            elif path == "/control":
                game.control(body)
                self._reply("{}")
            elif path == "/log":                  # the page's errors and connection state
                log(f"{self.client_address[0]} PAGE {json.dumps(body)[:2000]}")
                self._reply("{}")
            else:
                self.send_error(404)

    return Handler


def newest_model():
    runs = sorted(ROOT.glob("runs/*/last.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no trained model found (runs/*/last.pt); pass one: --vs runs/<run>/last.pt")
    return runs[-1]


def need_torch():
    """agents need torch: restart with the shared venv's python if this one has none"""
    try:
        import torch  # noqa: F401
    except ImportError:
        if VENV_PY.exists() and pathlib.Path(sys.prefix).resolve() != VENV_PY.parent.parent.resolve():   # not in the venv yet
            print(f"(no torch in {sys.executable}: restarting with {VENV_PY})", flush=True)
            os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)
        raise SystemExit(f"agents need torch: run this with {VENV_PY}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kv", nargs="*", help="battle keys, e.g. loc=Gemgrab1 hero=ShotgunGirl seed=1 diff=0")
    ap.add_argument("--vs", nargs="?", const="newest", metavar="MODEL",
                    help="the enemy team is played by this checkpoint (no value: the newest runs/*/last.pt)")
    ap.add_argument("--mates", metavar="MODEL|bots",
                    help="your two teammates: a checkpoint, or 'bots' (default: the --vs model, else bots)")
    ap.add_argument("--lan", action="store_true",
                    help="also serve on the local network (play on a phone in the same Wi-Fi; anyone there can join)")
    ap.add_argument("--serial", default="R52MB0DXE9B")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    battle = dict(s.split("=", 1) for s in a.kv)
    battle.setdefault("loc", "Gemgrab1")
    battle.setdefault("seed", "1")
    foes = newest_model() if a.vs == "newest" else a.vs
    mates = None if a.mates == "bots" else a.mates or foes
    for side, path in (("enemies", foes), ("teammates", mates)):
        if path:
            if not pathlib.Path(path).exists():
                raise SystemExit(f"model not found: {path}")
            print(f"{side}: agent {path}")
    if foes or mates:
        need_torch()

    print("starting the battle driver on the tablet ...")
    env = BrawlEnv(serial=a.serial, repeat=1, record=True)
    session = Session(env, battle, mates=mates, foes=foes)
    game = Game(session)
    loop = threading.Thread(target=game.run, daemon=True)
    loop.start()
    server = Server(("0.0.0.0" if a.lan else "127.0.0.1", a.port), make_handler(game))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{a.port}/"
    print(f"play at {url}  (Ctrl+C to quit)")
    if a.lan:
        for ip in lan_addresses():
            print(f"  on a phone in the same network: http://{ip}:{a.port}/")
        print("  (phone over USB instead: adb -s <phone serial> reverse tcp:%d tcp:%d, then open the 127.0.0.1 URL)"
              % (a.port, a.port))

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)   # also save the replay when stopped with kill
    if not a.no_open and not a.lan:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        game.stop()
        loop.join(timeout=5)
        session.save()
        env.close()


def lan_addresses():
    """this machine's IPv4 addresses in local networks (hostname -I), without asking anything outside"""
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ip for ip in out if ip.count(".") == 3 and not ip.startswith("127.") and not ip.startswith("172.17.")]


if __name__ == "__main__":
    main()
