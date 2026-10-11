#!/usr/bin/env python3
"""Batched client for the driver's `vec` mode: N offline bot battles in one driver process on the tablet, one binary
round trip per step for all of them (vs. one JSON round trip per battle-step in bsenv.BrawlEnv).

    from bsvec import VecDriver
    d = VecDriver(n=16, repeat=4)                      # starts the driver, builds 16 battles (seeds 1..16)
    obs = d.obs                                         # {"vec": float32 [n, obs_dim], "grid": uint8 [n, G, G]}
    act = np.zeros((d.n, d.act_dim), np.float32)        # rows: mx, my, attack, ax, ay, super, sx, sy, gadget, unused
    obs, rew, terminated, truncated, info = d.step(act)
    d.parts                                             # float32 [n, len(REWARD_PARTS)]: the reward pieces of that step
    d.view(obs["vec"], "enemy")["dx"]                   # named access: float32 [n, 3] (one column per enemy slot)
    d.close()

Actions are the fields of bsenv.Action: move direction (length < 0.3 = stand still), attack/super 0 none / 1 auto-aim /
2 aimed along (ax, ay) resp. (sx, sy) with length = fraction of range, gadget > 0.5. Buttons press on the first tick of
a step, movement is sent then (and every 16 ticks of long steps), so the agent decides (and the observation crosses USB)
once per `repeat` ticks (20 ticks = 1 s).
A battle that ends is rebuilt by the driver right away (seed += battles): that row has terminated/truncated set, info
holds the final tick/score/result of the finished battle, and obs is already the first observation of the next one.
Several brawlers per battle (protocol v5): VecDriver(n=96, agents=3) steers player 0 and its two teammates (the bots
keep only the enemy team): 32 battles, row r = brawler r % 3 of battle r // 3, each row with its own egocentric
observation, action and reward pieces (win / gem_diff / countdown are the team's, the rest the brawler's own).
Steps must cover whole battles (step_async ranges are multiples of agents).
info columns: INFO_FIELDS.

Reward: the driver reports separate pieces per step (d.parts, columns d.reward_names == REWARD_PARTS; definitions in
harness/jni/driver.c above rew_step), and reward = parts @ weights, with REWARD_WEIGHTS unless VecDriver(weights=...)
or d.set_weights(...) says otherwise (changeable at any time, no driver restart). Pieces, for the own brawler/team:
  win           +1 / -1 / 0 at game over                 gem_diff      change of (own - enemy team gems)
  countdown     change of the win-countdown potential    gems_carried  change of own carried gems
  kill, death   own kills / deaths this step             dmg_dealt     own damage to enemy brawlers / 1000
  dmg_taken     own hp lost in lives, weighted by how low hp already was (0.5 per life at full -> 1.5 near zero)
  low_hp        seconds spent alive below 10% hp
gem_diff and countdown are potential differences: over an episode they sum to the final gem difference and to +-1
when a countdown ended the game, so they guide without changing what winning means.

Observation (layout sent by the driver; d.blocks / d.view / d.describe()): egocentric float vector with blocks global,
self, ally x2, enemy x3, item x8, projectile x12, area x4, plus a GRID x GRID tile grid around the own hero, each cell
a bit set of GRID_FLAGS (blocks_move=1, blocks_shots=2, bush=4, destructible=8, outside=16, special=32).
Enemies hidden in bushes (fog of war, the game's own team-visibility check) keep their last seen position (visible=0,
seen_age > 0). Field meanings and scales: comment above OBS_BLOCKS in harness/jni/driver.c and docs/driver_plan.md.

Several drivers (one per tablet core): step_async() on all of them, then step_wait() as each one's reply arrives.
"""
import collections
import struct
import subprocess
import threading

import numpy as np

SERIAL = "R52MB0DXE9B"
DEV_DIR = "/data/local/tmp/bsh36"
MAGIC, VERSION = 0x31565342, 5
STEP, RESET, QUIT = 1, 2, 3
TERMINATED, TRUNCATED = 1, 2
INFO_FIELDS = ("flags", "tick", "score_own", "score_opp", "seed", "result")
ACT_FIELDS = ("mx", "my", "attack", "ax", "ay", "super", "sx", "sy", "gadget", "unused")
REWARD_PARTS = ("win", "gem_diff", "countdown", "gems_carried", "kill", "death", "dmg_dealt", "dmg_taken", "low_hp")
REWARD_WEIGHTS = {"win": 1.0, "gem_diff": 0.1, "countdown": 0.5, "gems_carried": 0.05, "kill": 0.05, "death": -0.1,
                  "dmg_dealt": 0.005, "dmg_taken": -0.1, "low_hp": -0.02}

Block = collections.namedtuple("Block", "name count fields start")   # start: offset in the obs vector


def tile_char(flags):
    """' ' outside the map, '#' wall (blocks shots), '~' water (blocks movement only), '"' bush, '+' special, '.' open"""
    return (" " if flags & 16 else "#" if flags & 2 else "~" if flags & 1 else '"' if flags & 4 else
            "+" if flags & 32 else ".")


def parse_layout(text):
    """'global:1:a,b;self:1:...;grid:17:flag,...' -> ({name: Block}, grid side, grid flag names)"""
    blocks, start, grid, flags = {}, 0, 0, ()
    for part in text.split(";"):
        name, count, fields = part.split(":", 2)
        if name == "grid":
            grid, flags = int(count), tuple(fields.split(","))
            continue
        b = Block(name, int(count), tuple(fields.split(",")), start)
        blocks[name] = b
        start += b.count * len(b.fields)
    return blocks, start, grid, flags


class VecDriver:
    def __init__(self, n=16, repeat=4, seed=1, serial=SERIAL, dev_dir=DEV_DIR, cpus=None, connect=True, weights=None,
                 out=None, **battle):
        """battle: driver keys (loc, hero, diff, skip=120, maxticks=12000, ...). cpus: optional list of tablet cores
        to pin the driver to. connect=False: return right after starting the driver process; call connect() later
        (lets several drivers load their data tables at the same time). weights: {piece: weight}, missing pieces keep
        their REWARD_WEIGHTS value. out: {"parts", "info", "vec", "grid"} -> C-contiguous arrays of shape [n, ...]
        that every batch is read straight into (no copies; use recv_batch() instead of step_wait())."""
        self._weights_arg, self._bound = weights, out
        self.serial, self.remote_pid = serial, None
        self.agents = int(battle.get("agents", 1))      # brawlers steered per battle: n rows = n / agents battles
        if n % self.agents:
            raise ValueError(f"n ({n}) must be a multiple of agents ({self.agents})")
        keys = {"n": n, "repeat": repeat, "seed": seed, **battle}
        if cpus is not None:   # best effort: the tablet isolates idle cores at times, then the driver runs unpinned
            keys["cpus"] = f"{sum(1 << c for c in cpus):x}"
        cmd = f"{dev_dir}/driver {dev_dir} {dev_dir}/assets vec " + " ".join(f"{k}={v}" for k, v in keys.items())
        # the shell prints its pid and exec's the driver, so the driver keeps that pid (for kill() on a hang)
        self.proc = subprocess.Popen(["adb", "-s", serial, "shell", "-T", f"su -c 'echo pid $$ >&2; exec {cmd}'"],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.log = collections.deque(maxlen=60)          # last stderr lines of the driver, for error messages
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._in, self._out = self.proc.stdin, self.proc.stdout
        self.n = None
        if connect:
            self.connect()

    def connect(self):
        """Wait for the driver's hello, the observation layout and the first observations."""
        if self.n is not None:
            return
        hello = np.frombuffer(self._read(8), np.int32)
        if hello[0] != MAGIC or hello[1] != VERSION:
            raise RuntimeError(f"unexpected driver hello {hello.tolist()} (expected protocol version {VERSION}; "
                               f"push the current harness/driver)")
        hello = np.frombuffer(self._read(36), np.int32)
        self.n, self.obs_dim, self.act_dim, self.info_dim, self.repeat, self.grid_size = (int(v) for v in hello[:6])
        self.layout = self._read(int(hello[6])).decode()
        self.reward_names = tuple(self._read(int(hello[8])).decode().split(","))
        if len(self.reward_names) != int(hello[7]):
            raise RuntimeError(f"reward piece names do not match their count: {self.reward_names}")
        self.set_weights(self._weights_arg)
        self.blocks, dim, grid, self.grid_flags = parse_layout(self.layout)
        if dim != self.obs_dim or grid != self.grid_size:
            raise RuntimeError(f"layout does not match the hello: {dim} vs {self.obs_dim}, grid {grid}")
        n, g = self.n, self.grid_size
        k = len(self.reward_names)
        floats = 4 + n + n * k + n * self.info_dim + n * self.obs_dim
        self._buf = bytearray(4 * floats + n * g * g)
        self._view = memoryview(self._buf)
        at = 0

        def take(dtype, count, shape=None):
            nonlocal at
            a = np.frombuffer(self._buf, dtype, count, at)
            at += a.nbytes
            return a.reshape(shape) if shape else a
        self._hdr = take(np.int32, 4)
        self._rew = take(np.float32, n)
        self._parts = take(np.float32, n * k, (n, k))
        self._info = take(np.int32, n * self.info_dim, (n, self.info_dim))
        self._obs = take(np.float32, n * self.obs_dim, (n, self.obs_dim))
        self._grid = take(np.uint8, n * g * g, (n, g, g))
        if self._bound is not None:
            want = {"parts": (np.float32, (n, k)), "info": (np.int32, (n, self.info_dim)),
                    "vec": (np.float32, (n, self.obs_dim)), "grid": (np.uint8, (n, g, g))}
            for name, (dtype, shape) in want.items():
                a = self._bound[name]
                if a.dtype != dtype or a.shape != shape or not a.flags.c_contiguous:
                    raise ValueError(f"out[{name!r}] must be C-contiguous {np.dtype(dtype).name} {shape}, got "
                                     f"{a.dtype} {a.shape}")
            self._parts, self._info = self._bound["parts"], self._bound["info"]
            self._obs, self._grid = self._bound["vec"], self._bound["grid"]
        self._targets = {}                               # (start, count) -> where the bytes of such a reply go
        self._inflight = collections.deque([(0, n)])     # battle ranges of the replies still to come, in order
        self.busy_us = self.reset_us = 0                 # driver time of the last batch (simulating / resetting)
        if self._bound is not None:
            self.recv_batch()
            self.obs = {"vec": self._obs, "grid": self._grid}
        else:
            self.obs = self._recv()[0]

    def set_weights(self, weights=None):
        """weights: {piece: weight}; pieces not named keep their REWARD_WEIGHTS value (unknown names raise)."""
        w = dict(REWARD_WEIGHTS)
        for name, v in (weights or {}).items():
            if name not in w:
                raise ValueError(f"unknown reward piece {name!r}; pieces: {', '.join(REWARD_PARTS)}")
            w[name] = float(v)
        self.weights = w
        if self.n is not None:
            missing = set(self.reward_names) - set(w)
            if missing:
                raise RuntimeError(f"driver reports reward pieces without a weight: {missing}")
            self._w = np.array([w[name] for name in self.reward_names], np.float32)

    # -- observation helpers --
    def view(self, vec, block):
        """Named fields of one block: {field: array [..., count]} (views into vec, no copy). vec: [..., obs_dim]."""
        b = self.blocks[block]
        nf = len(b.fields)
        part = vec[..., b.start:b.start + b.count * nf].reshape(*vec.shape[:-1], b.count, nf)
        return {f: part[..., i] for i, f in enumerate(b.fields)}

    def field_names(self):
        """One name per obs column, e.g. 'enemy1.dx'."""
        out = []
        for b in self.blocks.values():
            for k in range(b.count):
                out += [f"{b.name}{k if b.count > 1 else ''}.{f}" for f in b.fields]
        return out

    def describe(self, i=0, obs=None):
        """Readable dump of battle i's observation (for debugging)."""
        obs = self.obs if obs is None else obs
        lines = []
        for b in self.blocks.values():
            v = self.view(obs["vec"][i], b.name)
            for k in range(b.count):
                row = {f: float(v[f][k]) for f in b.fields}
                if b.count > 1 and not row.get("present", 1):
                    continue
                lines.append(f"{b.name}{k if b.count > 1 else ''}: " +
                             " ".join(f"{f}={x:.3g}" for f, x in row.items()))
        g, c = obs["grid"][i], self.grid_size // 2
        for y in range(self.grid_size):
            lines.append("".join("@" if (y, x) == (c, c) else tile_char(g[y, x]) for x in range(self.grid_size)))
        return "\n".join(lines)

    # -- plumbing --
    def _drain_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            line = line.decode(errors="replace").rstrip()
            if self.remote_pid is None and line.startswith("pid "):
                self.remote_pid = int(line[4:])
                continue
            self.log.append(line)

    def _dead(self, what):
        self.proc.poll()
        tail = "\n".join(list(self.log)[-15:])
        return RuntimeError(f"driver {what} (exit code {self.proc.returncode}); last driver output:\n{tail}")

    def _read(self, k):
        data = bytearray(k)
        self._readinto(memoryview(data))
        return bytes(data)

    def _readinto(self, mv):
        got = 0
        while got < len(mv):
            r = self._out.readinto(mv[got:])
            if not r:
                raise self._dead("closed its output")
            got += r

    def _write(self, data):
        try:
            mv, put = memoryview(data), 0
            while put < len(mv):
                put += self._in.write(mv[put:])
        except (BrokenPipeError, OSError):
            raise self._dead("is gone") from None

    def _range_targets(self, start, count):
        t = self._targets.get((start, count))
        if t is None:
            rows = slice(start, start + count)
            t = self._targets[start, count] = [self._view[:16 + 4 * count]] + [
                memoryview(a[rows]).cast("B") for a in (self._parts, self._info, self._obs, self._grid)]
        return t

    def recv_batch(self):
        """Read the next reply (the oldest step/reset in flight) into the driver's arrays (the out= arrays when
        given), without copying anything. -> (start, count): the battle rows it filled."""
        start, count = self._inflight.popleft()
        for mv in self._range_targets(start, count):
            self._readinto(mv)
        if self._hdr[0] != MAGIC or self._hdr[1] != count:
            raise RuntimeError(f"protocol out of sync: header {self._hdr.tolist()}")
        self.busy_us, self.reset_us = int(self._hdr[2]), int(self._hdr[3])
        return start, count

    def _recv(self):
        self.recv_batch()
        flags = self._info[:, 0]
        obs = {"vec": self._obs.copy(), "grid": self._grid.copy()}
        self.parts = self._parts.copy()
        rew = self.parts @ self._w
        return obs, rew, (flags & TERMINATED) != 0, (flags & TRUNCATED) != 0, self._info.copy()

    # -- API --
    def step_async(self, actions, repeat=0, start=0):
        """Send one action row per battle; repeat <= 0 uses the driver's default ticks per step. With start, the rows
        are for battles start .. start+len(actions)-1 only (several ranges may be in flight; replies come in order,
        read them with recv_batch())."""
        a = np.ascontiguousarray(actions, np.float32)
        c = len(a)
        if a.ndim != 2 or a.shape[1] != self.act_dim or c < 1 or start < 0 or start + c > self.n:
            raise ValueError(f"actions must have shape (count, {self.act_dim}) with start + count <= {self.n}, got "
                             f"{a.shape} at start {start}")
        self._write(struct.pack("<5i", MAGIC, STEP, int(repeat), c, int(start)) + a.tobytes())
        self._inflight.append((int(start), c))

    def step_wait(self):
        """-> obs {"vec": [n, obs_dim], "grid": [n, G, G]}, reward [n] (weighted pieces), terminated [n], truncated [n],
        info [n, len(INFO_FIELDS)]; the step's reward pieces are in self.parts [n, len(reward_names)]
        (for full steps: step_async without start)"""
        out = self._recv()
        self.obs = out[0]
        return out

    def step(self, actions, repeat=0):
        self.step_async(actions, repeat)
        return self.step_wait()

    def reset_async(self, seed=None):
        """send a reset; its reply (all n battles) comes after the steps already in flight (recv_batch())"""
        self._write(struct.pack("<4i", MAGIC, RESET, -1 if seed is None else int(seed), 0))
        self._inflight.append((0, self.n))

    def reset(self, seed=None):
        """Rebuild every battle: battle i gets seed + i, or its next seed in the stream when seed is None."""
        self.reset_async(seed)
        self.obs = self._recv()[0]
        return self.obs

    def kill(self):
        """Kill the driver (also on the tablet, in case it hangs) without the QUIT handshake."""
        if self.remote_pid is not None:
            try:
                subprocess.run(["adb", "-s", self.serial, "shell", f"su -c 'kill -9 {self.remote_pid}'"],
                               capture_output=True, timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        if self.proc.poll() is None:
            try:
                while self._inflight:
                    self.recv_batch()
                self._write(struct.pack("<4i", MAGIC, QUIT, 0, 0))
                self._in.close()
                self.proc.wait(timeout=10)
            except (RuntimeError, subprocess.TimeoutExpired, OSError):
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":   # smoke test: 8 battles, random actions until 8 episodes have ended
    import time
    rng = np.random.default_rng(0)
    with VecDriver(n=8) as d:
        print(f"n={d.n} obs_dim={d.obs_dim} grid={d.grid_size}x{d.grid_size} act_dim={d.act_dim} repeat={d.repeat}")
        print(d.describe(0))
        t0, steps, ends = time.perf_counter(), 0, []
        while len(ends) < 8:
            act = np.zeros((d.n, d.act_dim), np.float32)
            act[:, :2] = rng.uniform(-1, 1, (d.n, 2))
            act[:, 2] = rng.random(d.n) < 0.3
            act[:, 5] = 1
            obs, rew, term, trunc, info = d.step(act)
            steps += 1
            if steps == 100:
                print("--- battle 0 after 100 steps ---")
                print(d.describe(0))
            for i in np.flatnonzero(term | trunc):
                ends.append(dict(zip(INFO_FIELDS, info[i].tolist())))
        dt = time.perf_counter() - t0
        print(f"{steps} batches x {d.n} battles in {dt:.2f}s: {steps * d.n / dt:.0f} env steps/s, "
              f"{steps * d.n * d.repeat / dt:.0f} ticks/s")
        for e in ends:
            print(e)
