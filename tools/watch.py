#!/usr/bin/env python3
"""Spectator for agent games: serves the viewer page in watch mode and streams the frames of the battles a BrawlEnv
plays (training or inference) to it while they are being played.

    env = BrawlEnv(watch=True)            # or watch=8766 (port); prints the URL and opens the browser
    env = BrawlEnv(watch=True, pace=1.0)  # also slow the agent to real time (20 ticks/s) so it is watchable

The page follows the newest episode live; untick "follow live" to scrub/replay what was buffered, or pick an earlier
episode (the last `keep` episodes stay in memory). Frames carry the inputs the game received (move target, aim
point, auto-aim target), drawn on the map, and an optional per-step note from the agent: env.step(action,
note={"value": v, "reward": r}) shows up in the "Agent" card.
"""
import json
import pathlib
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from viewer_assets import send_gfx

VIEWER = pathlib.Path(__file__).resolve().parent / "viewer/viewer.html"


class Spectator:
    def __init__(self, port=8766, keep=10, open_browser=True, title="agent"):
        self.lock, self.episodes, self.next_id, self.keep, self.title = threading.Lock(), [], 0, keep, title
        page = VIEWER.read_text().replace("/*REPLAY_JSON*/", json.dumps({"watch": True, "title": title}))
        spectator = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, body, ctype="application/json"):
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path in ("/", "/index.html"):
                    self._reply(page, "text/html; charset=utf-8")
                elif send_gfx(self, url.path):
                    pass
                elif url.path == "/poll":
                    q = {k: int(v[0]) for k, v in parse_qs(url.query).items()}
                    self._reply(json.dumps(spectator.poll(q.get("ep", -1), q.get("from", 0), q.get("names", 0))))
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        print(f"[watch] spectate at {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    def begin(self, meta, names):
        """New episode. `names` is the env's growing name table (shared, indices stay valid)."""
        with self.lock:
            self.episodes.append({"id": self.next_id, "meta": meta, "frames": [], "names": names, "done": False})
            self.next_id += 1
            del self.episodes[:-self.keep]

    def add(self, frame, done=False):
        with self.lock:
            if self.episodes:
                self.episodes[-1]["frames"].append(frame)
                self.episodes[-1]["done"] = done

    def poll(self, ep, start, names_from):
        with self.lock:
            if not self.episodes:
                return {"episodes": []}
            e = next((e for e in self.episodes if e["id"] == ep), self.episodes[-1])
            names = e["names"]
            return {"episodes": [[x["id"], x["meta"]["loc"], x["meta"]["seed"], len(x["frames"]), x["done"]]
                                 for x in self.episodes],
                    "ep": e["id"], "meta": e["meta"], "done": e["done"], "frames": e["frames"][start:],
                    "names": [[i, names[i]] for i in range(names_from, len(names))]}

    def close(self):
        self.server.shutdown()
