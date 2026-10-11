#!/usr/bin/env python3
"""Training dashboard: charts of a run's metrics and the newest eval / training battles, one click to watch any of them.

    python3 tools/dashboard.py                    # newest run in runs/, opens http://localhost:8800
    python3 tools/dashboard.py runs/first20 --port 8801 --no-open

It is a separate process that only reads files the training writes anyway (runs/<run>/log.csv, eval/u*/s*.npz,
watch/*.npz), so it costs training nothing. The tablet is used only when you open a battle: it is re-simulated once
in serve mode (~3 s, one core) into a viewer page next to the .npz, and later clicks open that page directly.
"""
import argparse
import csv
import json
import pathlib
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from episodes import Episode  # noqa: E402
from replay import build_page  # noqa: E402
from viewer_assets import send_gfx  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
render_lock = threading.Lock()          # one re-simulation on the tablet at a time
_meta_cache = {}                         # npz path -> (mtime, summary)


def battle_summary(path):
    """result / score / length of one saved battle (cached; loading an .npz takes ~1 ms)"""
    mtime = path.stat().st_mtime
    hit = _meta_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    ep = Episode.load(path)
    s = {"file": str(path.relative_to(ROOT)), "seed": ep.seed, "result": ep.result, "score": list(ep.score),
         "seconds": round(ep.ticks / 20), "ret": round(ep.ret, 2), "time": mtime,
         "rendered": path.with_suffix(".html").exists()}
    _meta_cache[path] = (mtime, s)
    return s


def run_state(run):
    """everything the page shows, as JSON"""
    metrics = []
    log = run / "log.csv"
    if log.exists():
        with open(log, newline="") as f:
            for row in csv.DictReader(f):
                metrics.append({k: (float(v) if v not in ("", "nan") else None) for k, v in row.items()})
    evals = []
    for d in sorted((run / "eval").glob("u*"), reverse=True):
        battles = []
        for p in sorted(d.glob("s*.npz")):
            try:
                battles.append(battle_summary(p))
            except (OSError, ValueError, KeyError):
                pass                      # being written right now
        if battles:
            evals.append({"update": int(d.name[1:]), "battles": battles})
    watch = []
    for p in sorted((run / "watch").glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try:
            watch.append(battle_summary(p))
        except (OSError, ValueError, KeyError):
            pass
    runs = sorted((p.name for p in RUNS.iterdir() if (p / "log.csv").exists()), reverse=True) if RUNS.exists() else []
    try:                                   # written by train_ppo.py; runs before it steered player 0 only
        agents = int(json.loads((run / "config.json").read_text()).get("agents", 1))
    except (OSError, ValueError):
        agents = 1
    return {"run": run.name, "runs": runs, "metrics": metrics, "evals": evals, "watch": watch, "agents": agents,
            "baselines": {"gems": 27, "random": 0} if agents == 3 else {"gems": 72, "random": 17}}


class Handler(BaseHTTPRequestHandler):
    run = None

    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            self.send(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/state":
            run = RUNS / q["run"][0] if "run" in q else self.run
            if run.parent != RUNS or not run.is_dir():
                return self.send(404, "no such run", "text/plain")
            self.send(200, json.dumps(run_state(run)), "application/json")
        elif send_gfx(self, url.path):
            pass
        elif url.path == "/replay":
            path = (ROOT / q.get("f", [""])[0]).resolve()
            if RUNS.resolve() not in path.parents or path.suffix != ".npz" or not path.exists():
                return self.send(404, "no such battle", "text/plain")
            page = path.with_suffix(".html")
            if not page.exists():
                with render_lock:
                    if not page.exists():
                        # in a fresh process, so a dashboard left running always re-simulates with the current
                        # code (an old process once replayed 3-brawler battles with player 0's actions only)
                        r = subprocess.run([sys.executable, str(ROOT / "tools/episodes.py"), str(path), "--no-open"],
                                           capture_output=True, text=True, timeout=300)
                        if r.returncode or "diverged" in r.stderr:
                            page.unlink(missing_ok=True)
                            return self.send(500, "re-simulation failed:\n" + r.stderr[-3000:], "text/plain")
            if path.with_suffix(".json").exists():
                build_page(path.with_suffix(".json"))     # always with the current viewer (3D assets etc.)
            self.send(200, page.read_bytes(), "text/html; charset=utf-8")
        else:
            self.send(404, "not found", "text/plain")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Dashboard</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --fg:#1d2330; --muted:#6b7385; --line:#e3e6ec; --win:#1f9d55; --loss:#d64545;
        --draw:#8a8f9c; --c1:#2f6fde; --c2:#e08a1e; --c3:#8e44ad; }
@media (prefers-color-scheme: dark) { :root { --bg:#14171d; --card:#1d2129; --fg:#e6e8ee; --muted:#9097a6;
        --line:#2c313b; --win:#3ecf7a; --loss:#ff6b6b; --draw:#8a8f9c; --c1:#6ea2ff; --c2:#ffb054; --c3:#c58cf0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { display:flex; gap:16px; align-items:center; flex-wrap:wrap; padding:12px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0; }
select { font:inherit; background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:3px 6px; }
.muted { color:var(--muted); }
main { padding:16px; display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); max-width:1500px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; min-width:0; }
.card h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
canvas { width:100%; height:220px; display:block; }
.legend { display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:4px; }
.legend i { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:4px; }
.stats { display:grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap:8px; }
.stat b { display:block; font-size:18px; }
.stat span { font-size:11px; color:var(--muted); }
.quick { display:flex; gap:8px; flex-wrap:wrap; }
a.btn { display:inline-block; padding:6px 10px; border-radius:6px; border:1px solid var(--line); color:var(--fg); text-decoration:none; }
a.btn:hover { border-color:var(--c1); }
.quick a.b { color:#fff; border:none; padding:7px 11px; }
.battles { display:flex; flex-wrap:wrap; gap:5px; margin:4px 0 10px; }
.b { display:inline-block; min-width:52px; text-align:center; padding:3px 6px; border-radius:5px; font-size:12px;
     text-decoration:none; color:#fff; font-variant-numeric:tabular-nums; }
.b.w { background:var(--win); } .b.l { background:var(--loss); } .b.d { background:var(--draw); }
.b.r { outline:2px solid var(--fg); outline-offset:-2px; }
.scroll { max-height:420px; overflow:auto; }
.evhead { font-size:12px; color:var(--muted); }
</style></head><body>
<header><h1>Training Dashboard</h1>
  <label class="muted">run <select id="run"></select></label>
  <span class="muted" id="status"></span></header>
<main>
  <div class="card"><h2>Now</h2><div class="stats" id="stats"></div>
    <h2 style="margin-top:14px">Watch the newest</h2><div class="quick" id="quick"></div>
    <p class="muted" style="font-size:12px;margin:10px 0 0">A battle opens in a new tab. The first time, it is re-simulated
    on the tablet (~3 s); outlined ones are ready already.</p></div>
  <div class="card"><h2>Win rate</h2><canvas id="cwin"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>training games</span>
    <span><i style="background:var(--c2)"></i>eval (fixed battles)</span>
    <span><i style="background:var(--muted)"></i><span id="refs">gems heuristic 72% / random 17%</span></span></div></div>
  <div class="card"><h2>Gem difference (own − enemy)</h2><canvas id="cgem"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>training</span>
    <span><i style="background:var(--c2)"></i>eval</span></div></div>
  <div class="card"><h2>Learning health</h2><canvas id="chealth"></canvas>
    <div class="legend"><span><i style="background:var(--c1)"></i>value explained variance (0..1, higher = better)</span>
    <span><i style="background:var(--c3)"></i>entropy / its start (1 = fully random)</span></div></div>
  <div class="card"><h2>Eval battles (newest first)</h2><div class="scroll" id="evals"></div></div>
  <div class="card"><h2>Training games, battle 0 (newest first)</h2><div class="scroll" id="watch"></div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let run = new URLSearchParams(location.search).get("run");

function chart(canvas, series, opt = {}) {
  // series: [{xs, ys, color, dots}], opt: {lo, hi, refs: [values], pct}
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const pts = series.flatMap(s => s.xs.map((x, i) => [x, s.ys[i]])).filter(p => p[1] != null);
  if (!pts.length) { g.fillStyle = css("--muted"); g.fillText("no data yet", 10, 20); return; }
  const L = 44, R = 8, T = 8, B = 22;
  const x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]), x0 + 1);
  let lo = opt.lo ?? Math.min(...pts.map(p => p[1]), ...(opt.refs || [])), hi = opt.hi ?? Math.max(...pts.map(p => p[1]), ...(opt.refs || []));
  if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (hi - y) / (hi - lo) * (H - T - B);
  g.strokeStyle = css("--line"); g.fillStyle = css("--muted"); g.font = "11px system-ui"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillText(opt.pct ? Math.round(v) + "%" : v.toFixed(Math.abs(hi - lo) < 5 ? 2 : 1), 4, y + 4);
  }
  g.fillText((x0 / 1e6).toFixed(1) + "M", L, H - 6);
  const xl = (x1 / 1e6).toFixed(1) + "M decisions"; g.fillText(xl, W - R - g.measureText(xl).width, H - 6);
  for (const r of opt.refs || []) {
    g.setLineDash([4, 4]); g.strokeStyle = css("--muted");
    g.beginPath(); g.moveTo(L, Y(r)); g.lineTo(W - R, Y(r)); g.stroke(); g.setLineDash([]);
  }
  for (const s of series) {
    g.strokeStyle = g.fillStyle = s.color; g.lineWidth = 2; g.beginPath(); let on = false;
    s.xs.forEach((x, i) => { const y = s.ys[i]; if (y == null) { on = false; return; }
      if (s.dots) { g.moveTo(X(x) + 3.5, Y(y)); g.arc(X(x), Y(y), 3.5, 0, 7); }
      else { on ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); on = true; } });
    s.dots ? g.fill() : g.stroke();
  }
}

function smooth(ys, k = 5) {   // running mean over the last k values (training numbers jump from update to update)
  return ys.map((_, i) => { const w = ys.slice(Math.max(0, i - k + 1), i + 1).filter(v => v != null);
    return w.length ? w.reduce((a, b) => a + b, 0) / w.length : null; });
}

function battleLink(b, label) {
  const cls = b.result > 0 ? "w" : b.result < 0 ? "l" : "d";
  const a = document.createElement("a");
  a.className = "b " + cls + (b.rendered ? " r" : ""); a.target = "_blank";
  a.href = "/replay?f=" + encodeURIComponent(b.file);
  a.textContent = label ?? `${b.score[0]}–${b.score[1]}`;
  a.title = `seed ${b.seed}, ${b.seconds} s, return ${b.ret}` + (b.rendered ? " (ready)" : " (re-simulates ~3 s)");
  return a;
}

function fmt(v, d = 2, pct = false) { return v == null ? "–" : pct ? v.toFixed(1) + "%" : v.toFixed(d); }

async function refresh() {
  let s;
  try { s = await (await fetch("/api/state" + (run ? "?run=" + encodeURIComponent(run) : ""))).json(); }
  catch (e) { $("status").textContent = "dashboard server not reachable"; return; }
  run = s.run;
  const sel = $("run");
  if (sel.options.length !== s.runs.length) sel.innerHTML = s.runs.map(r => `<option>${r}</option>`).join("");
  sel.value = s.run;
  const M = s.metrics, xs = M.map(m => m.steps), last = M[M.length - 1] || {};
  const ev = M.filter(m => m.eval_win != null);
  $("status").textContent = M.length ? `${M.length} updates · updated ${new Date().toLocaleTimeString()}` : "waiting for the first update";
  const lastEval = ev[ev.length - 1] || {};
  const stats = [["decisions", last.steps != null ? (last.steps / 1e6).toFixed(2) + "M" : "–"],
    ["decisions/s", fmt(last.sps, 0)], ["train win (avg 5)", fmt(smooth(M.map(m => m.win)).pop(), 1, true)],
    ["last eval win", lastEval.eval_win != null ? fmt(100 * lastEval.eval_win, 1, true) : "–"],
    ["best eval win", ev.length ? fmt(100 * Math.max(...ev.map(m => m.eval_win)), 1, true) : "–"],
    ["eval gem diff", fmt(lastEval.eval_gdiff)], ["entropy", fmt(last.ent)], ["value ev", fmt(last.ev)]];
  $("stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");

  chart($("cwin"), [{xs, ys: smooth(M.map(m => m.win)), color: css("--c1")},
    {xs: ev.map(m => m.steps), ys: ev.map(m => 100 * m.eval_win), color: css("--c2"), dots: true},
    {xs: ev.map(m => m.steps), ys: ev.map(m => 100 * m.eval_win), color: css("--c2")}], {lo: 0, hi: 100, refs: [s.baselines.gems, s.baselines.random], pct: true});
  $("refs").textContent = `gems heuristic ${s.baselines.gems}% / random ${s.baselines.random}%` +
    (s.agents > 1 ? ` (all ${s.agents} brawlers steered)` : "");
  chart($("cgem"), [{xs, ys: smooth(M.map(m => m.gdiff)), color: css("--c1")},
    {xs: ev.map(m => m.steps), ys: ev.map(m => m.eval_gdiff), color: css("--c2"), dots: true}], {refs: [0]});
  chart($("chealth"), [{xs, ys: M.map(m => m.ev), color: css("--c1")},
    {xs, ys: M.map(m => m.ent == null ? null : m.ent / M[0].ent), color: css("--c3")}], {lo: 0, hi: 1});

  const quick = $("quick"); quick.innerHTML = "";
  const newest = s.evals[0];
  if (newest) {
    const w = newest.battles.find(b => b.result > 0), l = newest.battles.find(b => b.result < 0);
    if (w) quick.append(battleLink(w, `eval u${newest.update}: a win ${w.score[0]}–${w.score[1]}`));
    if (l) quick.append(battleLink(l, `eval u${newest.update}: a loss ${l.score[0]}–${l.score[1]}`));
  }
  if (s.watch[0]) quick.append(battleLink(s.watch[0], `newest training game ${s.watch[0].score[0]}–${s.watch[0].score[1]}`));
  if (!quick.children.length) quick.innerHTML = '<span class="muted">nothing recorded yet (evals every --eval-every updates; training games need --watch)</span>';
  for (const a of quick.children) a.classList.add("btn");

  const evals = $("evals"); evals.innerHTML = "";
  for (const e of s.evals) {
    const wins = e.battles.filter(b => b.result > 0).length;
    const h = document.createElement("div"); h.className = "evhead";
    h.textContent = `update ${e.update}: ${wins}/${e.battles.length} won`;
    const row = document.createElement("div"); row.className = "battles";
    e.battles.forEach(b => row.append(battleLink(b)));
    evals.append(h, row);
  }
  if (!s.evals.length) evals.innerHTML = '<span class="muted">no eval yet</span>';
  const watch = $("watch"); watch.innerHTML = "";
  const row = document.createElement("div"); row.className = "battles";
  s.watch.forEach(b => row.append(battleLink(b)));
  watch.append(row);
  if (!s.watch.length) watch.innerHTML = '<span class="muted">none (start training with --watch)</span>';
}

$("run").onchange = e => { run = e.target.value; history.replaceState(null, "", "?run=" + run); refresh(); };
addEventListener("resize", refresh);
refresh(); setInterval(refresh, 10000);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=pathlib.Path, help="run directory (default: the newest in runs/)")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    if a.run:
        run = a.run.resolve()
    else:
        runs = sorted(p for p in RUNS.glob("*") if (p / "log.csv").exists()) if RUNS.exists() else []
        if not runs:
            sys.exit("no runs in runs/ yet")
        run = max(runs, key=lambda p: (p / "log.csv").stat().st_mtime)
    if run.parent != RUNS.resolve():
        sys.exit(f"{run} is not a directory in {RUNS}")
    Handler.run = run
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/?run={run.name}"
    print(f"dashboard for {run.name}: {url}  (Ctrl-C to stop)")
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
