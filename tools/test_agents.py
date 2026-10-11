#!/usr/bin/env python3
"""Checks for steering several brawlers per battle (driver protocol v5, battle key agents=k). Tablet attached:
    python3 tools/test_agents.py            (~2 min)

1. control: with agents=3 and idle actions our two teammates stand still (the bots walk 10+ tiles in 10 s when
   agents=1 leaves them to the bot AI)
2. consistent views: with random actions, every step, the three rows of a battle agree with each other:
   teammate i as seen from row j (position, velocity, hp, gems, brawler) == row i's own values; enemy positions
   differ between rows exactly by the rows' offset; the team's reward pieces (win, gem_diff, countdown) and the
   info columns are the same in all three rows
3. replay: a finished 3-brawler episode re-simulates in serve mode to the same final score and tick
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import VecDriver  # noqa: E402
from episodes import EpisodeRecorder, rerender  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def ally_slot(i, j):
    """slot of agent i in agent j's ally block (allies are the other team members in player order)"""
    return [a for a in range(3) if a != j].index(i)


def test_control():
    moved = {}
    for agents in (1, 3):
        with VecDriver(n=4 * agents, agents=agents, seed=1) as d:
            A0 = {k: v.copy() for k, v in d.view(d.obs["vec"], "ally").items()}
            idle = np.zeros((d.n, d.act_dim), np.float32)
            for _ in range(50):                                   # 10 s
                obs, *_ = d.step(idle)
            A1 = d.view(obs["vec"], "ally")
            rows = np.arange(0, d.n, agents)                      # player 0's rows; player 0 stands still
            moved[agents] = 10 * np.hypot(A1["dx"][rows] - A0["dx"][rows], A1["dy"][rows] - A0["dy"][rows])
    check("control: bot teammates walk (agents=1)", (moved[1] > 3).mean() > 0.7, f"tiles {moved[1].round(1).ravel()}")
    check("control: steered teammates stand still when idle (agents=3)", (moved[3] < 0.05).all(),
          f"tiles {moved[3].round(2).ravel()}")


def test_views_and_replay():
    rng = np.random.default_rng(0)
    bad, checked, team_bad, eps = {}, 0, 0, []
    with VecDriver(n=24, agents=3, seed=3_000_000) as d:
        rec = EpisodeRecorder(d, loc="Gemgrab1", agents=3)
        obs = d.obs
        for t in range(700):
            vec = obs["vec"]
            S, A, E = d.view(vec, "self"), d.view(vec, "ally"), d.view(vec, "enemy")
            for b in range(0, d.n, 3):
                if not (S["alive"][b:b + 3] > 0).all():
                    continue                                      # positions of dead heroes are last-known ones
                checked += 1
                for j in range(3):                                # the viewing row
                    for i in range(3):                            # the teammate looked at
                        if i == j:
                            continue
                        s, r, rj = ally_slot(i, j), b + i, b + j
                        pairs = {"dx": (A["dx"][rj, s], -A["dx"][b + i, ally_slot(j, i)]),
                                 "dy": (A["dy"][rj, s], -A["dy"][b + i, ally_slot(j, i)]),
                                 "vx": (A["vx"][rj, s], S["vx"][r, 0]), "vy": (A["vy"][rj, s], S["vy"][r, 0]),
                                 "hp": (A["hp"][rj, s], S["hp"][r, 0]), "gems": (A["gems"][rj, s], S["gems"][r, 0]),
                                 "hero": (A["hero"][rj, s], S["hero"][r, 0])}
                        for k, (x, y) in pairs.items():
                            if abs(x - y) > 1e-4:
                                bad[k] = bad.get(k, 0) + 1
                    # enemies: the same last-seen positions, shifted by the offset between the rows
                    if j:
                        off = A["dx"][b, ally_slot(j, 0)], A["dy"][b, ally_slot(j, 0)]
                        seen = (E["present"][b] > 0) & (E["seen_age"][b] < 1)
                        dx = E["dx"][b] - E["dx"][b + j] - off[0]
                        dy = E["dy"][b] - E["dy"][b + j] - off[1]
                        if (np.abs(dx[seen]) > 1e-4).any() or (np.abs(dy[seen]) > 1e-4).any():
                            bad["enemy"] = bad.get("enemy", 0) + 1
            act = np.zeros((d.n, d.act_dim), np.float32)
            ang = rng.uniform(0, 2 * np.pi, d.n)
            act[:, 0], act[:, 1] = np.cos(ang), np.sin(ang)
            act[:, 2] = rng.integers(0, 2, d.n)
            act[:, 5] = 1
            obs, rew, term, trunc, info = d.step(act)
            parts = d.parts.reshape(-1, 3, d.parts.shape[1])
            team = [d.reward_names.index(k) for k in ("win", "gem_diff", "countdown")]
            I = info.reshape(-1, 3, info.shape[1])
            team_bad += int((np.abs(parts[:, :, team] - parts[:, :1, team]) > 1e-6).any(axis=(1, 2)).sum())
            team_bad += int((I != I[:, :1]).any(axis=(1, 2)).sum())
            eps += rec.step(act, rew, term, trunc, info)
    check("views: teammates seen from each row == their own values", not bad and checked > 1000,
          f"{checked} battle-steps checked, mismatches {bad}")
    check("views: team reward pieces and info equal in all rows", team_bad == 0, f"{team_bad} differing battle-steps")
    check("replay: finished 3-brawler episodes exist", len(eps) > 0, f"{len(eps)} finished")
    if eps:
        ep = eps[0]
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rerender(ep, pathlib.Path("scratch/replays/test_agents.json"))
        check("replay: re-simulation reaches the same score and tick", "diverged" not in err.getvalue(),
              f"seed {ep.seed} score {ep.score} tick {ep.ticks} actions {ep.actions.shape}")


if __name__ == "__main__":
    test_control()
    test_views_and_replay()
    print("ALL PASS" if not FAILS else f"FAILED: {', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
