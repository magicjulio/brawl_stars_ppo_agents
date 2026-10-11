#!/usr/bin/env python3
"""Example agent loop with the live spectator: a hand-written chase-and-shoot policy standing in for a trained
model. Shows how inference plugs into BrawlEnv and how per-step notes reach the viewer.

  python3 tools/example_agent.py                       # watch it play at real time in the browser
  python3 tools/example_agent.py --episodes 3 --fast   # full speed (follow-live skips ahead), no pacing
  python3 tools/example_agent.py --no-watch --save     # headless; save replays to scratch/replays/
"""
import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsenv import Action, BrawlEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def policy(obs, meta):
    """Walk at the closest enemy hero until it is within 70% of attack range; auto-aim attack and super."""
    me = obs.own
    enemies = [o for o in obs.objects if o.type == "brawler" and o.team != (me.team if me else 0) and o.hp > 0]
    if not me or not enemies:
        return Action(), {}
    e = min(enemies, key=lambda o: math.hypot(o.x - me.x, o.y - me.y))
    dx, dy = e.x - me.x, e.y - me.y
    d = math.hypot(dx, dy)
    rng = meta["range"][0]
    act = Action(move=(dx, dy) if d > 0.7 * rng else (0, 0), attack=Action.AUTO if d <= rng else Action.NONE,
                 super_=Action.AUTO)
    return act, {"target": e.name, "dist_tiles": d / 300, "in_range": int(d <= rng)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kv", nargs="*", help="battle keys, e.g. loc=Gemgrab1 hero=ShotgunGirl")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=4, help="ticks per decision")
    ap.add_argument("--fast", action="store_true", help="do not pace to real time")
    ap.add_argument("--no-watch", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="start the spectator without opening a browser")
    ap.add_argument("--save", action="store_true", help="save each episode's replay")
    a = ap.parse_args()
    battle = dict(s.split("=", 1) for s in a.kv)
    watch = False if a.no_watch else True
    env = BrawlEnv(repeat=a.repeat, record=a.save, watch=watch, watch_open=not a.no_open, pace=0 if a.fast else 1.0,
                   **battle)
    try:
        for ep in range(a.episodes):
            obs = env.reset(seed=int(battle.get("seed", 1)) + ep)
            ret, prev = 0, obs.score
            while not obs.done:
                act, info = policy(obs, env.meta)
                obs = env.step(act, note={**info, "return": ret})
                ret += (obs.score[0] - prev[0]) - (obs.score[1] - prev[1])   # toy reward: gem score difference
                prev = obs.score
            print(f"episode {ep}: tick {obs.tick} score {obs.score[0]}-{obs.score[1]} return {ret}")
            if a.save:
                path = ROOT / f"scratch/replays/agent_{env.meta['loc']}_s{env.meta['seed']}.json"
                env.save_replay(path)
                print(f"  replay: {path} (tools/replay.py --file {path.relative_to(ROOT)})")
        if env.spectator:
            try:
                input("done - press Enter to stop the spectator ")
            except EOFError:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
