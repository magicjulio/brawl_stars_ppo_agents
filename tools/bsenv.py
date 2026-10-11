#!/usr/bin/env python3
"""Step-wise control of an offline bot battle running headless on the tablet (driver `serve` mode).

    from bsenv import BrawlEnv, Action
    env = BrawlEnv(repeat=4)                     # one decision = 4 ticks (0.2 s)
    obs = env.reset(loc="Gemgrab1", seed=1)
    while not obs.done:
        obs = env.step(Action(move=(0, -1), attack=Action.AUTO))
    env.close()

Watching / evaluation (all off by default, so training pays nothing for them):
    BrawlEnv(record=True)    keep the episode's frames -> env.save_replay("scratch/replays/x.json"), then
                             tools/replay.py --file scratch/replays/x.json builds the viewer page
    BrawlEnv(watch=True)     live spectator page following the episodes as they are played (tools/watch.py);
                             only stores references to frames the env parsed anyway, encoding happens on browser poll
    BrawlEnv(pace=1.0)       throttle to real time (20 ticks/s) so a watched agent is watchable; 0 = full speed
    env.step(action, note={"value": v, "reward": r})   per-step numbers shown in the viewer's "Agent" card

Action space (continuous, as the in-game controls):
    move    (x, y)  direction; length < 0.3 = stand still. +x = right, +y = down (towards the blue spawn in Gem Grab)
    attack  0 none | 1 auto-aim (closest visible enemy in range; else fires where the hero faces) | 2 aimed
    aim     (x, y)  aim direction for attack=2; length 0..1 = fraction of max range (only throwers use the length)
    super_  / super_aim  same as attack / aim, ignored until the super is charged
    gadget  bool    (the default loadout has no gadget yet)
Attack/super/gadget are presses on the first tick of a step; movement keeps steering for all `repeat` ticks.

Observation (Obs): tick, score (blue, red), objects (list of Obj; heroes carry their gem count), grid (tile names,
row-major w*h), own hero state (ammo, ammo_max, super_charge, super_needed, alive), Gem Grab win countdown (ticks left,
leading team), inputs the game received this step, done.
Positions are logic units: 300 per tile, 20 ticks/s.
env.meta after reset: loc, hero, seed, w, h (tiles), players, range = [attack range, super range] in logic units.
"""
import json
import subprocess
import time
from dataclasses import dataclass, field

SERIAL = "R52MB0DXE9B"
DEV_DIR = "/data/local/tmp/bsh36"
TYPES = ("brawler", "projectile", "area_effect", "item")


@dataclass
class Action:
    NONE, AUTO, AIM = 0, 1, 2
    move: tuple = (0.0, 0.0)
    attack: int = 0
    aim: tuple = (0.0, 0.0)
    super_: int = 0
    super_aim: tuple = (0.0, 0.0)
    gadget: bool = False

    def fields(self):
        f = lambda v: repr(float(v))     # exact: float32 actions replay bit-identically (tools/episodes.py)
        return [f(self.move[0]), f(self.move[1]), str(int(self.attack)), f(self.aim[0]), f(self.aim[1]),
                str(int(self.super_)), f(self.super_aim[0]), f(self.super_aim[1]), str(int(bool(self.gadget)))]

    def wire(self, repeat, *more):
        """the serve-mode step line; more: Actions of the further rows (reset keys agents=k, enemies=m);
        None = no input for that brawler this step"""
        return " ".join(["step", str(int(repeat))] + [x for a in (self,) + more for x in (a.fields() if a else ["-"])])


@dataclass
class Obj:
    gid: int
    type: str
    name: str
    x: int
    y: int
    z: int
    team: int
    player: int     # player index for heroes, -1 otherwise
    hp: int         # -1 for non-characters
    max_hp: int
    gems: int = -1  # gems carried (heroes only; -1 for everything else)


@dataclass
class Obs:
    tick: int
    score: tuple
    objects: list
    grid: list
    ammo: float          # shots available (fractional while reloading)
    ammo_max: int
    super_charge: int
    super_needed: int
    alive: bool
    countdown: int       # Gem Grab: ticks until the leading team wins (0 = no countdown running)
    countdown_team: int  # team that wins when it runs out, -1 none
    inputs: list         # [[type, world x, world y, auto-aim target gid]]: 0 attack, 1 super, 2 move, 3 stop, 8 gadget
    done: bool
    raw: list = field(repr=False, default=None)
    reward_parts: dict = field(default_factory=dict)   # reward pieces of the step that led here (bsvec.REWARD_PARTS)

    @property
    def own(self):
        return next((o for o in self.objects if o.player == 0 and o.type == "brawler"), None)

    def reward(self, weights=None):
        """Weighted sum of reward_parts, as bsvec.VecDriver computes it (weights: overrides of REWARD_WEIGHTS)."""
        from bsvec import REWARD_WEIGHTS
        w = {**REWARD_WEIGHTS, **(weights or {})}
        return sum(w[k] * v for k, v in self.reward_parts.items())


class BrawlEnv:
    def __init__(self, repeat=4, serial=SERIAL, dev_dir=DEV_DIR, record=False, watch=False, watch_open=True, pace=0.0,
                 **battle):
        """battle: default reset keys (loc, hero, seed, diff, ...), as for the driver's m2 command.
        watch: False, True or a port number for the live spectator page (watch_open: open it in the browser).
        pace: 1.0 = real time, 0 = full speed."""
        self.repeat, self.record, self.battle, self.pace = repeat, record, battle, pace
        self.spectator, self._clock = None, None
        if watch:
            from watch import Spectator
            self.spectator = Spectator(port=8766 if watch is True else int(watch), open_browser=watch_open)
        cmd = f"{dev_dir}/driver {dev_dir} {dev_dir}/assets serve"
        self.proc = subprocess.Popen(["adb", "-s", serial, "shell", f"su -c '{cmd}'"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.names, self.meta, self.grid, self.frames = [], None, [], []
        msg = self._read()
        if not msg.get("ready"):
            raise RuntimeError(f"driver did not start: {msg}")

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"driver exited (code {self.proc.poll()})")
            if line.startswith("{"):
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self._read()

    def _obs(self, msg, note=None):
        for i, n in msg.get("names", []):
            while len(self.names) <= i:
                self.names.append("")
            self.names[i] = n
        f = msg["frame"]
        if note is not None:
            f.append(note)
        if f[4]:
            self.grid = [self.names[i] for i in f[4]]
        if self.record:
            self.frames.append(f)
        if self.spectator:
            self.spectator.add(f, bool(msg["done"]))
        tick, s0, s1, objs, _, own, inputs = f[:7]
        parts = dict(zip(self.meta["rc_names"].split(","), msg["rc"])) if "rc" in msg and self.meta else {}
        return Obs(tick=tick, score=(s0, s1), grid=self.grid, done=bool(msg["done"]), raw=f, inputs=inputs,
                   reward_parts=parts,
                   objects=[Obj(o[0], TYPES[o[1]] if 0 <= o[1] < 4 else str(o[1]), self.names[o[2]], *o[3:11]) for o in objs],
                   ammo=own[0] / 1000, ammo_max=own[1], super_charge=own[2], super_needed=own[3], alive=own[4] != 0,
                   countdown=own[5] if len(own) > 5 else 0, countdown_team=own[6] if len(own) > 6 else -1)

    def reset(self, **battle):
        """New battle. skip=120 (default) fast-forwards the start countdown, during which inputs have no effect."""
        kw = {"skip": 120, **self.battle, **battle}
        msg = self._send("reset " + " ".join(f"{k}={v}" for k, v in kw.items()))
        self.meta, self.frames, self._clock = msg["meta"], [], None
        if self.spectator:
            self.spectator.begin(self.meta, self.names)
        return self._obs(msg)

    def step(self, action=None, repeat=None, note=None):
        """action: an Action, or a list of them (one per row: player 0, its steered teammates (reset key agents=k), then
        the steered enemies (enemies=m), as env.meta["rows"]); None in the list = no input for that row this step.
        note: optional dict of numbers (value estimate, reward, ...) stored with the frame for the viewer."""
        repeat = repeat or self.repeat
        if self.pace > 0:   # real-time pacing for watching; never used in training
            now = time.perf_counter()
            self._clock = max(self._clock or now, now - 0.5)
            time.sleep(max(0.0, self._clock - now))
            self._clock += repeat / (20 * self.pace)
        acts = list(action) if isinstance(action, (list, tuple)) else [action or Action()]
        return self._obs(self._send(acts[0].wire(repeat, *acts[1:])), note)

    def observe(self):
        """The policy observation of every steered row (env.meta["rows"]), as vec mode / training builds it (team 1
        sees the battle turned around, see harness/jni/driver.c MIRROR_*): (vec float32 [rows, obs_dim],
        grid uint8 [rows, G, G]); layout: env.meta["obs_layout"] (bsvec.parse_layout). Velocities are measured since
        the previous call, so call it once per decision (training: every 4 ticks)."""
        import numpy as np
        msg = self._send("obs")
        g = int(round((len(msg["grid"][0]) // 2) ** 0.5)) if msg["grid"] else 0   # hex: 2 characters per tile
        grid = np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in msg["grid"]], np.uint8)
        return np.array(msg["obs"], np.float32), grid.reshape(len(msg["grid"]), g, g)

    def save_replay(self, path):
        with open(path, "w") as fh:
            json.dump({"meta": self.meta, "frames": self.frames, "names": self.names}, fh, separators=(",", ":"))

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":   # smoke test: walk up, fire with auto-aim, report timing
    with BrawlEnv(repeat=1, record=True) as env:
        obs = env.reset(seed=1)
        t0, n = time.perf_counter(), 0
        first_move = None
        while not obs.done and n < 400:
            obs = env.step(Action(move=(0, -1), attack=Action.AUTO if n % 10 == 0 else 0))
            n += 1
            if first_move is None and obs.own and obs.own.y != 9750:
                first_move = obs.tick
        dt = time.perf_counter() - t0
        print(f"{n} steps in {dt:.2f}s = {1e3 * dt / n:.1f} ms/step round trip; first movement at tick {first_move}")
        print(f"tick {obs.tick} score {obs.score} own {obs.own} ammo {obs.ammo:.2f}/{obs.ammo_max} "
              f"super {obs.super_charge}/{obs.super_needed}")
