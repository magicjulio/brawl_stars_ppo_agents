#!/usr/bin/env python3
"""Checks that trained agents can play the enemy team (driver serve mode, battle key enemies=m; tools/play.py --vs).
Tablet attached, torch needed:
    ../../.venv/bin/python tools/test_vs_agents.py [--model runs/team45/last.pt] [--battles 16]     (~5 min)

1. control: with enemies=3 and no input the enemy team stands still (the bots would walk)
2. mirror: at the start the enemy team sees the battle turned around: the enemy brawler on the opposite spawn sees
   the same own position, gem mine offset and tile grid as player 0
3. strength, with the eval battles' seeds (1_000_000 + k), every agent deciding every 4 ticks:
   a) red = 3 agents vs blue = 1 agent (player 0) + 2 bots: printed only (team45: red 44%; outcomes follow the seed's
      lineup: the same seeds were blowouts for the same side in a) and b))
   b) 3 agents vs 3 agents: no side should dominate (expect ~50% +- noise); broken mirroring would make red lose most.
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsenv import Action, BrawlEnv  # noqa: E402
from bsvec import parse_layout  # noqa: E402
from play import Policy, newest_model  # noqa: E402

FAILS = []
DECIDE = 4


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def heroes(obs):
    return {o.player: o for o in obs.objects if o.type == "brawler"}


def play(env, policy, seed, agents, enemies):
    """one battle with every steered row played by the policy -> (blue gems, red gems, ticks)"""
    obs = env.reset(loc="Gemgrab1", seed=seed, agents=agents, enemies=enemies)
    while not obs.done:
        vec, grid = env.observe()
        obs = env.step(policy.act(vec, grid), repeat=DECIDE)
    return obs.score[0], obs.score[1], obs.tick


def strength(env, policy, battles, agents, enemies):
    res = []
    for k in range(battles):
        b, r, t = play(env, policy, 1_000_000 + k, agents, enemies)
        res.append(np.sign(b - r))
        print(f"    seed {1_000_000 + k}: blue {b} - red {r}  (tick {t})", flush=True)
    res = np.array(res)
    return (res < 0).mean(), (res > 0).mean(), (res == 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--battles", type=int, default=16)
    a = ap.parse_args()
    model = a.model or newest_model()
    print(f"model {model}")
    with BrawlEnv(repeat=DECIDE) as env:
        # 1. control
        obs = env.reset(loc="Gemgrab1", seed=1, agents=1, enemies=3)
        start = heroes(obs)
        for _ in range(50):
            obs = env.step(Action(), repeat=4)            # rows left out stand still
        end = heroes(obs)
        moved = [np.hypot(end[p].x - start[p].x, end[p].y - start[p].y) / 300 for p in env.meta["rows"][1:]]
        check("control: steered enemies stand still without input", max(moved) == 0, f"tiles {np.round(moved, 1)}")

        # 2. mirror
        obs = env.reset(loc="Gemgrab1", seed=1, agents=1, enemies=3)
        blocks = parse_layout(env.meta["obs_layout"])[0]
        vec, grid = env.observe()
        h, rows = heroes(obs), env.meta["rows"]
        W, H = env.meta["w"] * 300, env.meta["h"] * 300
        partner = next(r for r, p in enumerate(rows)
                       if h[p].team == 1 and (W - h[p].x, H - h[p].y) == (h[0].x, h[0].y))
        cols = [blocks["self"].start + blocks["self"].fields.index(f) for f in ("x", "y")] + \
               [blocks["global"].start + blocks["global"].fields.index(f) for f in ("mine_dx", "mine_dy")]
        same = np.allclose(vec[0, cols], vec[partner, cols]) and (grid[0] == grid[partner]).all()
        check("mirror: the enemy on the opposite spawn sees what player 0 sees", same,
              f"player 0 {np.round(vec[0, cols], 3)} vs player {rows[partner]} {np.round(vec[partner, cols], 3)}")

        # 3. strength
        policy = Policy(model, env.meta["obs_layout"])
        print(f"  a) red = 3 agents vs blue = agent + 2 bots, {a.battles} battles")
        red, blue, draw = strength(env, policy, a.battles, agents=1, enemies=3)
        print(f"INFO  red agents vs agent + 2 bots: red wins {100 * red:.0f}%, blue {100 * blue:.0f}%, draws "
              f"{100 * draw:.0f}% (2026-10-10, team45: 44%; the lineup a seed draws decides most battles)")
        print(f"  b) 3 agents vs 3 agents, {a.battles} battles")
        red, blue, draw = strength(env, policy, a.battles, agents=3, enemies=3)
        check("strength: agents vs agents is balanced", abs(red - blue) <= 0.5,
              f"red wins {100 * red:.0f}%, blue {100 * blue:.0f}%, draws {100 * draw:.0f}%")
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
