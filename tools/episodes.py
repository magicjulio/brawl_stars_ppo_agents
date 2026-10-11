#!/usr/bin/env python3
"""Episode logs for vec battles, and turning any logged episode into a viewer replay afterwards.

Training never records frames: a battle is fully determined by its seed, battle keys and the agent's actions (the
vec battle is tick-identical to the serve-mode battle, tools/test_vec.py), so the recorder keeps only the action rows
(40 bytes per battle-step) plus optional per-step numbers (reward, value, ...). Any finished episode can then be
re-simulated in serve mode with recording on and opened in the viewer, with those numbers on its timeline.

    rec = EpisodeRecorder(d, loc="Gemgrab1")             # d: a VecDriver; same battle keys as the driver got
    obs = d.obs
    while training:
        act = policy(obs)
        obs, rew, term, trunc, info = d.step(act)
        for ep in rec.step(act, rew, term, trunc, info, value=v):   # finished episodes (usually none)
            ep.save(f"scratch/episodes/{ep.seed}.npz")
    tools/episodes.py scratch/episodes/123.npz           # -> scratch/replays/ep_Gemgrab1_s123.html, opens it

Cost when off: nothing (it is a separate object). When on: one [n, 10] copy per step plus one float per extra column.
Battles with several steered brawlers (battle key agents=k): an episode holds the actions of all k (row t = their k
action rows side by side, [T, 10k]) so it re-simulates exactly; its columns are those of the recorded row.
"""
import argparse
import dataclasses
import json
import pathlib
import sys
import webbrowser

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import INFO_FIELDS, REWARD_PARTS, TERMINATED  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Episode:
    seed: int
    battle: dict            # driver battle keys (loc, hero, skip, ...), without seed
    repeat: np.ndarray      # int16 [T] ticks per step (the driver default unless the step overrode it)
    actions: np.ndarray     # float32 [T, 10 * agents] exactly as sent (agents = battle.get("agents", 1))
    columns: dict           # name -> float32 [T]: reward plus any extras given to EpisodeRecorder.step
    result: int             # +1 win / -1 loss / 0 draw (0 also when truncated)
    score: tuple            # final (own, opponent) gems
    ticks: int              # final game tick
    terminated: bool

    @property
    def ret(self):
        return float(self.columns["reward"].sum())

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("repeat", "actions", "columns")}
        np.savez_compressed(path, meta=json.dumps(meta), repeat=self.repeat, actions=self.actions,
                            **{f"col_{k}": v for k, v in self.columns.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        meta["score"] = tuple(meta["score"])
        cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
        return cls(repeat=z["repeat"], actions=z["actions"], columns=cols, **meta)


class EpisodeRecorder:
    """Keeps the actions of every running battle of one VecDriver and returns each episode when it ends."""

    def __init__(self, driver, **battle):
        self.n, self.default_repeat = driver.n, driver.repeat
        self.battle = {k: v for k, v in battle.items() if k not in ("seed", "n", "repeat", "cpus")}
        self.k = int(self.battle.get("agents", 1))          # one episode per battle, from its first row
        self.seed = None                       # current seed per battle, learned from info at the first step
        self.acts = [[] for _ in range(self.n)]
        self.reps = [[] for _ in range(self.n)]
        self.cols = [{} for _ in range(self.n)]

    def step(self, actions, rew, term, trunc, info, repeat=0, **extras):
        """Call after every driver step with what went in and came out; extras: name -> array [n] (value, ...).
        Returns the list of Episodes that ended at this step."""
        actions = np.asarray(actions, np.float32)
        rep = repeat if repeat > 0 else self.default_repeat
        done = []
        for i in range(0, self.n, self.k):
            self.acts[i].append(actions[i:i + self.k].ravel())
            self.reps[i].append(rep)
            c = self.cols[i]
            c.setdefault("reward", []).append(float(rew[i]))
            for k, v in extras.items():
                c.setdefault(k, []).append(float(v[i]))
            if term[i] or trunc[i]:
                f = dict(zip(INFO_FIELDS, info[i].tolist()))
                done.append(Episode(seed=f["seed"], battle=dict(self.battle), repeat=np.array(self.reps[i], np.int16),
                                    actions=np.stack(self.acts[i]), ticks=f["tick"],
                                    columns={k: np.array(v, np.float32) for k, v in c.items()},
                                    result=f["result"], score=(f["score_own"], f["score_opp"]),
                                    terminated=bool(f["flags"] & TERMINATED)))
                self.acts[i], self.reps[i], self.cols[i] = [], [], {}
        return done

    def reset(self):
        """Call after VecDriver.reset(): the battles in progress were thrown away."""
        self.acts = [[] for _ in range(self.n)]
        self.reps = [[] for _ in range(self.n)]
        self.cols = [{} for _ in range(self.n)]


class EnvRecorder:
    """Records whole episodes of chosen battles of a BrawlVecEnv (tools/bsvecenv.py) through its on_step hook, for
    watching what the agent does during training at no real cost (one action row per watched battle-step).

        rec = EnvRecorder(env, env_ids=[0, 300], out_dir="scratch/watch")   # attach; saves every finished episode
        ...training...                                  # rec.episodes: the last `keep` finished Episodes
        tools/episodes.py scratch/watch/e0_s10000032.npz  # -> viewer replay, re-simulated exactly
        rec.close()                                     # detach

    extras: {name: fn(env_id array) -> float array} sampled per step (e.g. the value estimate), stored as columns.
    Only episodes watched from their first step are kept; a battle lost in a driver crash is dropped."""

    def __init__(self, env, env_ids=(0,), out_dir=None, keep=20, extras=None):
        self.env, self.out_dir, self.keep, self.extras = env, out_dir, keep, extras or {}
        self.ids = np.asarray(sorted(set(int(i) for i in env_ids)))
        self.mask = np.zeros(env.num_envs, bool)
        self.mask[self.ids] = True
        # a battle is watched from its start: now if the env has not stepped yet, else after its current episode
        fresh = not env._started
        self.live = {int(i): fresh for i in self.ids}
        self.buf = {int(i): {"acts": [], "cols": {}} for i in self.ids}
        self.episodes = []
        env.on_step.append(self)

    def close(self):
        if self in self.env.on_step:
            self.env.on_step.remove(self)

    def __call__(self, env, env_id, actions, obs, rew, term, trunc, info):
        sel = np.flatnonzero(self.mask[env_id])
        if not len(sel):
            return
        ids = env_id[sel]
        extra = {k: np.asarray(fn(ids), np.float32) for k, fn in self.extras.items()}
        K = self.env.agents
        for j, (r, i) in enumerate(zip(sel, ids)):
            i = int(i)
            b = self.buf[i]
            if self.live[i]:
                r0 = r - i % K                                  # the battle's first row (groups hold whole battles)
                b["acts"].append(np.array(actions[r0:r0 + K], np.float32).ravel())
                c = b["cols"]
                c.setdefault("reward", []).append(float(rew[r]))
                for k, name in enumerate(REWARD_PARTS):
                    c.setdefault(name, []).append(float(info["parts"][r, k]))
                for k, v in extra.items():
                    c.setdefault(k, []).append(float(v[j]))
            if term[r] or trunc[r]:
                if self.live[i] and not info["crashed"][r]:
                    self._finish(i, {f: int(info[f][r]) for f in INFO_FIELDS}, b)
                self.live[i] = True
                self.buf[i] = {"acts": [], "cols": {}}

    def _finish(self, i, f, b):
        env = self.env
        ep = Episode(seed=f["seed"], battle={k: v for k, v in env.battle[i // env.n].items()
                                             if k not in ("seed", "n", "repeat", "cpus")},
                     repeat=np.full(len(b["acts"]), env.repeat, np.int16), actions=np.stack(b["acts"]),
                     columns={k: np.array(v, np.float32) for k, v in b["cols"].items()}, result=f["result"],
                     score=(f["score_own"], f["score_opp"]), ticks=f["tick"],
                     terminated=bool(f["flags"] & TERMINATED))
        self.episodes = (self.episodes + [ep])[-self.keep:]
        if self.out_dir:
            ep.save(pathlib.Path(self.out_dir) / f"e{i}_s{ep.seed}.npz")


def to_action(row):
    """vec action row -> bsenv.Action, converted exactly as the driver's vec mode does (buttons rounded and clamped)"""
    from bsenv import Action
    btn = lambda v: int(min(2, max(0, np.rint(v))))
    r = [float(v) for v in row]
    return Action(move=(r[0], r[1]), attack=btn(r[2]), aim=(r[3], r[4]), super_=btn(r[5]), super_aim=(r[6], r[7]),
                  gadget=r[8] > 0.5)


def rerender(ep, out=None, serial=None, check=True):
    """Re-simulate an Episode in serve mode with recording on -> replay .json + viewer .html; returns the html path.
    Each frame carries the step's columns (reward, value, ...) as its note, so the viewer plots them."""
    from bsenv import BrawlEnv
    import replay
    kw = {"serial": serial} if serial else {}
    battle = {"skip": 120, **ep.battle}
    k = int(battle.get("agents", 1))
    with BrawlEnv(record=True, **kw) as env:
        obs = env.reset(seed=ep.seed, **battle)
        ret = 0.0
        for t in range(len(ep.actions)):
            note = {k: float(v[t]) for k, v in ep.columns.items()}
            ret += note.get("reward", 0.0)
            note["return"] = ret
            acts = [to_action(a) for a in ep.actions[t].reshape(k, -1)]
            obs = env.step(acts, repeat=int(ep.repeat[t]), note=note)
            if obs.done:
                break
        env.meta["episode"] = {"seed": ep.seed, "result": ep.result, "score": list(ep.score), "steps": len(ep.actions),
                               "columns": sorted(ep.columns) + ["return"]}
        if check and (tuple(obs.score) != tuple(ep.score) or obs.tick != ep.ticks):
            print(f"warning: re-simulation diverged: score {obs.score} tick {obs.tick} vs logged {ep.score} tick "
                  f"{ep.ticks} (different driver build or battle keys?)", file=sys.stderr)
        loc = battle.get("loc", "Gemgrab1")
        out = pathlib.Path(out) if out else ROOT / f"scratch/replays/ep_{loc}_s{ep.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        env.save_replay(out)
    return replay.build_page(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", type=pathlib.Path, help="episode .npz files")
    ap.add_argument("--serial")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    for path in a.episodes:
        ep = Episode.load(path)
        page = rerender(ep, path.with_suffix(".json"), a.serial)
        print(f"{path}: seed {ep.seed} result {ep.result:+d} score {ep.score[0]}-{ep.score[1]} -> {page}")
        if not a.no_open:
            webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
