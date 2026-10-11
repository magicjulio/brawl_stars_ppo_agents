#!/usr/bin/env python3
"""Evaluate a policy against the bots on a fixed set of Gem Grab battles (vec mode, full speed).

  tools/eval.py                                   # all baseline policies, 64 battles each, comparison table
  tools/eval.py --policy gems --episodes 200      # one policy
  tools/eval.py --policy mypkg.agent:act          # your own: act(obs, d) -> float32 [d.n, 10]
  tools/eval.py --policy gems --render 2          # also replay the first 2 losses and 2 wins in the viewer
  tools/eval.py --policy gems agents=3            # our whole team steered by the policy (one row per brawler)

Eval battles use seeds 1_000_000 + k (k < episodes), apart from training seeds, so every run of the same policy
sees the same battles. Every episode is logged (tools/episodes.py: seed + actions, a few KB) to
scratch/eval/<policy>/, so any of them can be replayed later:  tools/episodes.py scratch/eval/gems/s1000007.npz

From a training loop:  from eval import evaluate;  stats = evaluate(policy_fn, episodes=64)
stats["return"] is the mean episode return under the given reward weights (bsvec.REWARD_WEIGHTS by default) and
stats["reward_parts"] the mean episode sum of each reward piece; the logged episodes carry every piece as a column,
so the viewer can plot them.
"""
import argparse
import importlib
import json
import math
import pathlib
import sys
import time
import webbrowser

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import INFO_FIELDS, REWARD_PARTS, VecDriver  # noqa: E402
from episodes import EpisodeRecorder, rerender  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_SEED = 1_000_000
BATTLE_TICKS = 4320          # Gem Grab: a battle that ends earlier was decided by the gem countdown


# ---- baseline policies on the vec observation (act(obs, d) -> [n, 10]) ----
def idle(obs, d):
    return np.zeros((d.n, d.act_dim), np.float32)


_rng = np.random.default_rng(0)


def random_policy(obs, d):
    a = np.zeros((d.n, d.act_dim), np.float32)
    a[:, :2] = _rng.uniform(-1, 1, (d.n, 2))
    a[:, 2] = _rng.random(d.n) < 0.3
    a[:, 5] = 1
    return a


def _nearest_enemy(d, vec):
    """-> (dx, dy, dist, visible) of the nearest living enemy we know a position of, per battle (dist inf if none)"""
    E = d.view(vec, "enemy")
    dist = np.hypot(E["dx"], E["dy"])
    dist = np.where((E["present"] > 0) & (E["alive"] > 0), dist, np.inf)
    k = dist.argmin(1)
    r = np.arange(len(vec))
    return E["dx"][r, k], E["dy"][r, k], dist[r, k], E["visible"][r, k] > 0


def chase(obs, d):
    """Walk at the nearest enemy until within 70% of attack range; auto-aim attack in range, super when ready.
    Without a known enemy, walk to the gem mine."""
    vec = obs["vec"]
    S, G = d.view(vec, "self"), d.view(vec, "global")
    dx, dy, dist, vis = _nearest_enemy(d, vec)
    rng = S["range"][:, 0]
    a = np.zeros((d.n, d.act_dim), np.float32)
    has = np.isfinite(dist)
    far = has & (dist > 0.7 * rng)
    a[:, 0] = np.where(far, dx, np.where(has, 0, G["mine_dx"][:, 0]))
    a[:, 1] = np.where(far, dy, np.where(has, 0, G["mine_dy"][:, 0]))
    a[:, 2] = (vis & (dist <= rng)).astype(np.float32)
    a[:, 5] = 1
    return a


def gems(obs, d):
    """Gem Grab heuristic: pick up the nearest gem in sight, otherwise wait next to the gem mine; fight (auto-aim)
    whatever comes in range. Once the team holds 10+ and leads, carriers walk back towards their spawn (+y)."""
    vec = obs["vec"]
    S, G, I = d.view(vec, "self"), d.view(vec, "global"), d.view(vec, "item")
    dx, dy, dist, vis = _nearest_enemy(d, vec)
    rng = S["range"][:, 0]
    idist = np.where(I["present"] > 0, np.hypot(I["dx"], I["dy"]), np.inf)
    k = idist.argmin(1)
    r = np.arange(d.n)
    has_gem = np.isfinite(idist[r, k])
    tx = np.where(has_gem, I["dx"][r, k], G["mine_dx"][:, 0])
    ty = np.where(has_gem, I["dy"][r, k], G["mine_dy"][:, 0])
    near_mine = ~has_gem & (np.hypot(tx, ty) < 0.15)
    winning = (G["own_score"][:, 0] >= 1.0) & (G["own_score"][:, 0] > G["enemy_score"][:, 0])
    retreat = winning & (S["gems"][:, 0] > 0)
    a = np.zeros((d.n, d.act_dim), np.float32)
    a[:, 0] = np.where(retreat, 0, np.where(near_mine, 0, tx))
    a[:, 1] = np.where(retreat, 1, np.where(near_mine, 0, ty))
    a[:, 2] = (vis & (dist <= rng)).astype(np.float32)
    a[:, 5] = 1
    return a


POLICIES = {"idle": idle, "random": random_policy, "chase": chase, "gems": gems}


def load_policy(spec):
    if spec in POLICIES:
        return POLICIES[spec]
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "act")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def evaluate(policy, episodes=64, n=16, repeat=4, seed=EVAL_SEED, log_dir=None, quiet=False, weights=None, **battle):
    """Play `episodes` battles (seeds seed .. seed+episodes-1) with policy(obs, d) -> actions; returns a stats dict
    (and the logged Episodes under "episode_list" when log_dir is None)."""
    battle = {"loc": "Gemgrab1", **battle}
    n = min(n, episodes)                   # battles at a time; with agents=k the policy gets k rows per battle
    k = int(battle.get("agents", 1))
    eps, t0, steps = {}, time.perf_counter(), 0
    with VecDriver(n=n * k, repeat=repeat, seed=seed, weights=weights, **battle) as d:
        rec, obs = EpisodeRecorder(d, **battle), d.obs
        while len(eps) < episodes:
            act = policy(obs, d)
            obs, rew, term, trunc, info = d.step(act)
            steps += 1
            for ep in rec.step(act, rew, term, trunc, info, **dict(zip(d.reward_names, d.parts.T))):
                if seed <= ep.seed < seed + episodes:
                    eps[ep.seed] = ep
                    if log_dir:
                        ep.save(pathlib.Path(log_dir) / f"s{ep.seed}.npz")
            if not quiet and steps % 200 == 0:
                print(f"  {len(eps)}/{episodes} battles done", end="\r", flush=True)
    dt = time.perf_counter() - t0
    E = [eps[s] for s in sorted(eps)]
    res = np.array([e.result for e in E])
    own = np.array([e.score[0] for e in E])
    opp = np.array([e.score[1] for e in E])
    secs = np.array([e.ticks for e in E]) / 20
    w, l, dr = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    lo, hi = wilson(w, len(E))
    stats = {"episodes": len(E), "win": w, "loss": l, "draw": dr, "win_rate": w / len(E), "win_rate_ci95": [lo, hi],
             "score": (w + 0.5 * dr) / len(E), "gems_own": float(own.mean()), "gems_opp": float(opp.mean()),
             "gem_diff": float((own - opp).mean()), "seconds": float(secs.mean()),
             "decided_by_countdown": float((np.array([e.ticks for e in E]) < BATTLE_TICKS).mean()),
             "truncated": int(sum(not e.terminated for e in E)), "wall_seconds": dt, "battle": battle,
             "seed": seed, "repeat": repeat, "return": float(np.mean([e.ret for e in E])),
             "reward_parts": {k: float(np.mean([e.columns[k].sum() for e in E])) for k in REWARD_PARTS},
             "weights": d.weights}
    if not log_dir:
        stats["episode_list"] = E
    return stats


def fmt(name, s):
    lo, hi = s["win_rate_ci95"]
    return (f"{name:<14} {s['episodes']:>4}  {s['win']:>3}/{s['draw']:>3}/{s['loss']:<3}  {100 * s['win_rate']:5.1f}% "
            f"[{100 * lo:4.1f}-{100 * hi:5.1f}]  {s['gems_own']:5.2f} {s['gems_opp']:5.2f} {s['gem_diff']:+6.2f}  "
            f"{s['seconds']:6.1f}s  {100 * s['decided_by_countdown']:5.1f}%  {s['return']:+7.3f}  {s['wall_seconds']:5.1f}s")


HEADER = (f"{'policy':<14} {'n':>4}  {'W/D/L':<11}  {'win':>6} {'95% CI':<12}  {'own':>5} {'opp':>5} {'diff':>6}  "
          f"{'length':>7}  {'cd-end':>6}  {'return':>7}  {'wall':>6}")


def fmt_parts(name, s):
    """mean episode sum of each reward piece, and (in brackets) its weighted contribution to the return"""
    w = s["weights"]
    return f"{name:<14} " + "  ".join(f"{k} {v:+.2f} ({w[k] * v:+.2f})" for k, v in s["reward_parts"].items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kv", nargs="*", help="battle keys, e.g. hero=ShotgunGirl diff=3 (loc defaults to Gemgrab1)")
    ap.add_argument("--policy", default="idle,random,chase,gems", help="comma list; names or module:function")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--n", type=int, default=32, help="battles per driver")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    ap.add_argument("--render", type=int, default=0, help="replay the first K losses and K wins of each policy")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    battle = dict(s.split("=", 1) for s in a.kv)
    print(f"eval: {a.episodes} battles per policy, seeds {a.seed}..{a.seed + a.episodes - 1}, repeat {a.repeat}, "
          f"{battle or 'Gemgrab1 defaults'}")
    print("W/D/L wins/draws/losses; own/opp mean final gems; cd-end = battles decided by the gem countdown")
    rows, parts = [], []
    for spec in a.policy.split(","):
        out = ROOT / "scratch/eval" / spec.replace(":", "_").replace("/", "_")
        s = evaluate(load_policy(spec), a.episodes, a.n, a.repeat, a.seed, log_dir=out, **battle)
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        rows.append(fmt(spec, s))
        parts.append(fmt_parts(spec, s))
        print(" " * 40, end="\r")
        if len(rows) == 1:
            print(HEADER)
        print(rows[-1])
        if a.render:
            from episodes import Episode
            E = [Episode.load(p) for p in sorted(out.glob("s*.npz"))]
            pick = [e for e in E if e.result < 0][:a.render] + [e for e in E if e.result > 0][:a.render]
            for e in pick:
                page = rerender(e, out / f"s{e.seed}.json")
                print(f"  {'win ' if e.result > 0 else 'loss'} seed {e.seed} {e.score[0]}-{e.score[1]}: {page}")
                if not a.no_open:
                    webbrowser.open(page.resolve().as_uri())
    print("reward pieces, mean per battle (weighted contribution to the return):")
    print("\n".join(parts))
    print(f"logs: scratch/eval/<policy>/ (summary.json, s<seed>.npz -> tools/episodes.py s<seed>.npz)")


if __name__ == "__main__":
    main()
