#!/usr/bin/env python3
"""BrawlVecEnv: many offline bot battles behind one batched, gymnasium-style vector env (numpy only).

    from bsvecenv import BrawlVecEnv
    env = BrawlVecEnv(drivers=8, n=96, repeat=4, loc="Gemgrab1")   # 8 driver processes x 96 battles = 768 envs
    obs, info = env.reset()                    # obs {"vec": float32 [N, 223], "grid": uint8 [N, 21, 21]}
    obs, rew, term, trunc, info = env.step(actions)                 # actions float32 [N, 10] (bsvec.ACT_FIELDS)
    env.close()

Asynchronous use (the fast way; EnvPool style): every group of battles (half a driver: pipeline=2) runs on its own,
the policy acts on whichever groups have answered:
    obs, info = env.reset(); env.send(policy(obs))                  # all groups
    while training:
        obs, rew, term, trunc, info = env.recv()                    # the rows of every group that is ready (>= 1)
        env.send(policy(obs), info["env_id"])                       # actions for exactly those rows
Stepping all drivers in lockstep (step()) is ~3x slower: every step one random driver's reply comes ~40 ms late through
USB/adb (measured: 16 ms simulation, 63 ms median lockstep step), and lockstep waits for it each time.
Single groups: step_async(actions, group) / step_wait(group); env.group_slice(g) = the rows of group g (group g =
driver g // pipeline, its battles from (g % pipeline) * n/pipeline on).
Pipelining: each driver has its two halves in flight, so it simulates one while the other's reply, policy call and
actions are under way (driver protocol v4: a step may address any range of battles).

Speed (Gem Grab, repeat 4, async, random actions; "policy" = a fake policy taking that long per call, GIL released):
  drivers x n     policy 0 ms   2 ms    5 ms    10 ms
  8 x 32          12.2k         8.4k    6.9k            env steps/s (decisions), no pipelining
  8 x 96          16.3k        15.3k   15.1k   13.3k    no pipelining
  8 x 96, pipe 2  16.1k                15.4k   14.5k    <- default: ~65k ticks/s, ~30 games/s with random play
  16 x 32         12.8k        10.0k                    (more drivers than the tablet's 8 cores does not help)
The tablet's own ceiling (8 drivers simulating, no USB at all) is ~67-74k ticks/s whatever n is, so the env runs at
~93% of it: the game simulation (mostly the bots' A* pathfinding) is the limit, not USB or the host.
72 MB per driver at n=128.

Each driver has one persistent host I/O thread (socket reads release the GIL; one host loop reading the drivers in
turn halves throughput). Replies are read straight into env-wide arrays, no per-driver copies.

A battle that ends is rebuilt at once (auto-reset): its row has terminated/truncated set, info holds the finished
battle's numbers, obs is the first observation of the next battle. info (arrays over the returned rows):
  env_id           int [rows] which battles these rows are (row order of obs, reward, ...)
  parts [rows, 9]  reward pieces of this step (bsvec.REWARD_PARTS); reward = parts @ weights (env.set_weights anytime)
  flags tick score_own score_opp seed result   the driver's info columns (bsvec.INFO_FIELDS)
  final            terminated | truncated
  episode_return, episode_length   for final rows: the finished battle's return (under the weights at each step) and
                   number of steps; 0 elsewhere
  crashed          rows truncated because their driver died or hung and was restarted (reward 0, obs = new battle)
Truncation only happens at the driver's maxticks (12000, far above Gem Grab's 4320-tick limit) or by a crash, so
there is no final observation: bootstrap truncated rows from the previous step's value.

Seeds: driver k plays seeds seed + k*SEED_BLOCK + ... (auto-reset adds n per battle slot), so drivers never share a
battle and training (seed >= 10_000_000 by default) never plays eval battles (eval.EVAL_SEED = 1_000_000 ..).
A restarted driver continues with seeds above every seed it used, so a battle that crashed the driver is not retried.

Crashes: a driver that exits, breaks the protocol or does not answer within `timeout` seconds is killed (also on the
tablet) and restarted with the same settings; its battles come back truncated + crashed, the other drivers go on.
More than `max_restarts` restarts within `restart_window` seconds (e.g. the tablet is gone) raise RuntimeError.
env.crashes lists what happened (time, driver, seeds, reason, last driver output).

Hooks (step 6, recorders): env.on_step.append(fn), fn(env, env_id, actions, obs, rew, term, trunc, info) once per
finished group step (arguments are views, valid during the call); nothing runs when the list is empty.

  tools/bsvecenv.py --bench                                # throughput, random actions, async recv/send
  tools/bsvecenv.py --bench --mode sync async --policy-ms 0 2
"""
import pathlib
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import (ACT_FIELDS, DEV_DIR, INFO_FIELDS, REWARD_PARTS, REWARD_WEIGHTS, SERIAL, TERMINATED,  # noqa: E402
                   TRUNCATED, VecDriver)

SEED_BLOCK = 10_000_000          # seeds per driver
TRAIN_SEED = 10_000_000          # default base seed; eval battles live at 1_000_000 + k
ACT_LOW = np.array([-1, -1, 0, -1, -1, 0, -1, -1, 0, 0], np.float32)
ACT_HIGH = np.array([1, 1, 2, 1, 1, 2, 1, 1, 1, 1], np.float32)
_COL = {name: i for i, name in enumerate(INFO_FIELDS)}


class _Worker:
    """One driver: the main thread writes its requests (send_*), this worker's thread reads the replies in order.
    Every finished reply puts (key, exception or None) on the env's completion queue; key = the group for steps,
    ("start", k) / ("reset", k) otherwise."""

    def __init__(self, env, k):
        self.env, self.k = env, k
        self.rows = slice(k * env.n, (k + 1) * env.n)
        self.out = {name: a[self.rows] for name, a in (("parts", env._parts), ("info", env._info),
                                                         ("vec", env._vec), ("grid", env._grid))}
        self.groups = range(k * env.pipeline, (k + 1) * env.pipeline)
        self.d = None
        self.seed = env.seed + k * SEED_BLOCK
        self.max_seed = self.seed - 1        # highest seed this driver slot has played
        self.lock = threading.Lock()         # the driver object and the order of requests vs. expected replies
        self.expected = queue.Queue()        # (kind, key) per request written, in order
        self.crash_pending = {}              # group -> info of its battles lost in a restart while the host held it
        self.thread = threading.Thread(target=self._run, name=f"driver{k}", daemon=True)
        self.thread.start()

    def _start(self, seed):
        e = self.env
        self.d = VecDriver(n=e.n, repeat=e.repeat, seed=seed, serial=e.serial, dev_dir=e.dev_dir,
                           cpus=e.cpus[self.k] if e.cpus is not None else None, out=self.out, **e.battle[self.k])
        self._seen()

    def _seen(self):
        self.max_seed = max(self.max_seed, int(self.out["info"][:, _COL["seed"]].max()))

    # -- main thread --
    def send_step(self, group, actions, repeat, start):
        with self.lock:
            try:
                self.d.step_async(actions, repeat, start)
            except (RuntimeError, OSError):
                pass                          # the driver is gone: reading the reply fails and restarts it
            self.expected.put(("step", group))

    def send_reset(self, seed):
        with self.lock:
            try:
                self.d.reset_async(seed)
            except (RuntimeError, OSError):
                pass
            self.expected.put(("reset", ("reset", self.k)))

    def send_quit(self):
        self.expected.put(("quit", None))

    # -- worker thread --
    def _run(self):
        done = self.env._done
        try:
            self._start(self.seed)
            done.put((("start", self.k), None))
        except Exception as exc:                                   # noqa: BLE001  (reported to the main thread)
            done.put((("start", self.k), exc))
            return
        while True:
            kind, key = self.expected.get()
            if kind == "quit":
                break
            try:
                try:
                    self.d.recv_batch()
                    self._seen()
                except Exception as exc:                           # noqa: BLE001
                    if self._restart(exc, (kind, key)):
                        break
                    continue
                if kind == "step":
                    lost = self.crash_pending.pop(key, None)
                    if lost is not None:
                        self._crash_rows(key, lost)
                    self.env._crashed[key] = lost is not None
                done.put((key, None))
            except Exception as exc:                               # noqa: BLE001
                done.put((key, exc))
        with self.lock:
            if self.d is not None:
                self.d.close()

    def _crash_rows(self, group, info):
        rows = self.env.group_slice(group)
        info = info.copy()
        info[:, _COL["flags"]] = TRUNCATED
        info[:, _COL["result"]] = 0
        self.env._parts[rows] = 0
        self.env._info[rows] = info

    def _restart(self, exc, job):
        """The driver died or hung: replace it. Every request in flight completes now as truncated crash rows of the
        battles it was for; groups the host holds get theirs on their next step. -> True if a quit was pending."""
        e = self.env
        with self.lock:
            d = self.d
            old = e._last_info.copy()                          # last good numbers of the battles that were running
            tail = list(d.log)[-15:] if d is not None else []
            if d is not None:
                d.kill()
            e._note_crash(self.k, old[self.rows, _COL["seed"]].tolist(), exc, tail)
            self.d = None
            self._start(self.max_seed + 1)
            jobs = [job]
            while True:
                try:
                    jobs.append(self.expected.get_nowait())
                except queue.Empty:
                    break
        in_flight = {key for kind, key in jobs if kind == "step"}
        for g in self.groups:
            if g in in_flight:
                self._crash_rows(g, old[e.group_slice(g)])
                e._crashed[g] = True
                e._done.put((g, None))
            else:
                self.crash_pending[g] = old[e.group_slice(g)]
        for kind, key in jobs:
            if kind == "reset":                                # the new driver's battles are fresh anyway
                for g in self.groups:
                    self.crash_pending.pop(g, None)
                e._done.put((key, None))
        return any(kind == "quit" for kind, _ in jobs)


class BrawlVecEnv:
    def __init__(self, drivers=8, n=96, repeat=4, seed=TRAIN_SEED, pipeline=2, weights=None, cpus=None,
                 timeout=30.0, start_timeout=180.0, max_restarts=8, restart_window=600.0, copy=True, serial=SERIAL,
                 dev_dir=DEV_DIR, battle=None, **battle_keys):
        """drivers x n rows (battles x agents). battle keys (loc="Gemgrab1", hero=..., diff=..., maxticks=...,
        agents=k: brawlers we steer per battle, see bsvec) go to every driver;
        battle=[{...}, ...] (one dict per driver) gives drivers different battles (merged over the keyword keys).
        pipeline: groups per driver (each n/pipeline battles); the driver simulates one group while the others'
        replies and actions cross USB. cpus: one list of tablet cores per driver (or None). copy=False: single-group
        results are views of the env's buffers, valid until that group's next step. weights: {piece: weight} over
        bsvec.REWARD_WEIGHTS."""
        if n % pipeline:
            raise ValueError(f"n ({n}) must be a multiple of pipeline ({pipeline})")
        self.agents = int(battle_keys.get("agents", 1))
        if (n // pipeline) % self.agents:
            raise ValueError(f"n / pipeline ({n // pipeline}) must be a multiple of agents ({self.agents}): "
                             f"a group steps whole battles")
        if seed + drivers * SEED_BLOCK >= 2 ** 31:
            raise ValueError("seed range exceeds int32")
        if cpus is not None and len(cpus) != drivers:
            raise ValueError("cpus: one list of cores per driver")
        self.drivers, self.n, self.repeat, self.seed, self.pipeline = drivers, n, repeat, seed, pipeline
        self.groups = drivers * pipeline
        self.num_envs = N = drivers * n
        self.group_size = n // pipeline
        self.serial, self.dev_dir, self.cpus, self.copy = serial, dev_dir, cpus, copy
        self.timeout, self.start_timeout = timeout, start_timeout
        self.max_restarts, self.restart_window = max_restarts, restart_window
        base = {"loc": "Gemgrab1", **battle_keys}
        self.battle = [dict(base, **(battle[k] if battle else {})) for k in range(drivers)]
        self.on_step = []
        self.crashes = []
        self._lock = threading.Lock()
        # the shapes are fixed by the driver build; checked against the first driver's hello below
        self.obs_dim, self.act_dim, self.grid_size, self.n_parts = 223, len(ACT_FIELDS), 21, len(REWARD_PARTS)
        self._parts = np.zeros((N, self.n_parts), np.float32)
        self._info = np.zeros((N, len(INFO_FIELDS)), np.int32)
        self._vec = np.zeros((N, self.obs_dim), np.float32)
        self._grid = np.zeros((N, self.grid_size, self.grid_size), np.uint8)
        self._last_info = np.zeros_like(self._info)            # info of the battles currently running (for crashes)
        self._ep_ret = np.zeros(N, np.float64)
        self._ep_len = np.zeros(N, np.int64)
        self._env_id = np.arange(N)
        self.set_weights(weights)
        self._pending = [False] * self.groups
        self._crashed = [False] * self.groups
        self._acts = [None] * self.groups
        self._done = queue.Queue()                              # (key, exception or None) per finished reply
        now = time.monotonic()
        self._busy = {("start", k): now + start_timeout for k in range(drivers)}   # key -> deadline
        self._started = False
        self._closed = False
        self._workers = []
        try:
            self._workers = [_Worker(self, k) for k in range(drivers)]     # they load in parallel
            self._wait(lambda: not self._busy)
        except BaseException:
            self.close()
            raise
        d = self._workers[0].d
        if (d.obs_dim, d.act_dim, d.grid_size, d.reward_names) != (self.obs_dim, self.act_dim, self.grid_size,
                                                                   REWARD_PARTS):
            self.close()
            raise RuntimeError("driver shapes differ from what BrawlVecEnv expects")
        self.blocks, self.layout, self.reward_names = d.blocks, d.layout, d.reward_names
        self._last_info[:] = self._info

    # -- configuration --
    def set_weights(self, weights=None):
        """{piece: weight}; pieces not named keep their REWARD_WEIGHTS value. Takes effect at the next step."""
        w = dict(REWARD_WEIGHTS)
        for name, v in (weights or {}).items():
            if name not in w:
                raise ValueError(f"unknown reward piece {name!r}; pieces: {', '.join(REWARD_PARTS)}")
            w[name] = float(v)
        self.weights = w
        self._w = np.array([w[p] for p in REWARD_PARTS], np.float32)

    def group_slice(self, group):
        return slice(group * self.group_size, (group + 1) * self.group_size)

    def view(self, vec, block):
        """named fields of one observation block (see bsvec.VecDriver.view)"""
        return self._workers[0].d.view(vec, block)

    # -- gymnasium spaces (only when gymnasium is installed) --
    @property
    def single_observation_space(self):
        from gymnasium import spaces
        g = self.grid_size
        return spaces.Dict({"vec": spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32),
                            "grid": spaces.Box(0, 255, (g, g), np.uint8)})

    @property
    def single_action_space(self):
        from gymnasium import spaces
        return spaces.Box(ACT_LOW, ACT_HIGH, dtype=np.float32)

    @property
    def observation_space(self):
        from gymnasium.vector.utils import batch_space
        return batch_space(self.single_observation_space, self.num_envs)

    @property
    def action_space(self):
        from gymnasium.vector.utils import batch_space
        return batch_space(self.single_action_space, self.num_envs)

    # -- plumbing --
    def _note_crash(self, k, seeds, exc, tail):
        """called from worker k's thread; raises (-> the main thread) when drivers keep failing"""
        with self._lock:
            now = time.time()
            self.crashes.append({"time": now, "driver": k, "seeds": seeds, "reason": str(exc).splitlines()[0],
                                 "log": tail})
            recent = [c for c in self.crashes if now - c["time"] < self.restart_window]
        print(f"[BrawlVecEnv] driver {k} failed ({str(exc).splitlines()[0]}); restarting it "
              f"({len(recent)} restarts in {self.restart_window:.0f}s)", file=sys.stderr)
        if len(recent) > self.max_restarts:
            raise RuntimeError(f"{len(recent)} driver restarts within {self.restart_window:.0f}s, giving up; last: "
                               f"{exc}\n" + "\n".join(tail))

    def _driver_of(self, key):
        return key[1] if isinstance(key, tuple) else key // self.pipeline

    def _take(self, block):
        """Handle one finished reply (waiting for it when block). A driver past a deadline is killed: its thread
        then sees the broken pipe and restarts it. -> False when nothing finished (non-blocking)."""
        while True:
            if not self._busy:
                return False
            now = time.monotonic()
            due = min(self._busy.values())
            try:
                key, exc = self._done.get(timeout=max(0.0, due - now)) if block else self._done.get_nowait()
            except queue.Empty:
                if not block:
                    return False
                for k in sorted({self._driver_of(key) for key, t in self._busy.items() if t <= now}):
                    w = self._workers[k] if k < len(self._workers) else None
                    if w is None or w.d is None:
                        raise RuntimeError(f"driver {k} did not start in time")
                    print(f"[BrawlVecEnv] driver {k} did not answer in time, killing it", file=sys.stderr)
                    w.d.kill()
                    for key in self._busy:                     # give the restart time to load
                        if self._driver_of(key) == k:
                            self._busy[key] = now + self.start_timeout
                continue
            self._busy.pop(key, None)
            if exc is not None:
                raise exc
            return True

    def _wait(self, cond):
        while not cond():
            self._take(block=True)

    def _finish(self, group):
        """bookkeeping of a group whose step is complete -> views (obs, rew, term, trunc, info)"""
        self._pending[group] = False
        rows = self.group_slice(group)
        crashed = np.full(self.group_size, self._crashed[group])
        parts, inf = self._parts[rows], self._info[rows]
        rew = parts @ self._w
        flags = inf[:, 0]
        term, trunc = (flags & TERMINATED) != 0, (flags & TRUNCATED) != 0
        final = term | trunc
        ret, ln = self._ep_ret[rows], self._ep_len[rows]
        ret += rew
        ln += 1
        ep_ret, ep_len = np.where(final, ret, 0.0), np.where(final, ln, 0)
        ret[final] = 0
        ln[final] = 0
        self._last_info[rows] = inf      # (rows that just finished keep the old battle's numbers until next step)
        obs = {"vec": self._vec[rows], "grid": self._grid[rows]}
        info = {name: inf[:, i] for i, name in enumerate(INFO_FIELDS)}
        info.update(env_id=self._env_id[rows], parts=parts, final=final, episode_return=ep_ret,
                    episode_length=ep_len, crashed=crashed)
        for fn in self.on_step:
            fn(self, info["env_id"], self._acts[group], obs, rew, term, trunc, info)
        return obs, rew, term, trunc, info

    def _out(self, results):
        """one group's results (copied unless copy=False) or several concatenated"""
        if len(results) == 1:
            obs, rew, term, trunc, info = results[0]
            if self.copy:
                obs = {k: v.copy() for k, v in obs.items()}
                info = {k: v.copy() for k, v in info.items()}
            return obs, rew, term, trunc, info
        obs = {k: np.concatenate([r[0][k] for r in results]) for k in results[0][0]}
        rew, term, trunc = (np.concatenate([r[i] for r in results]) for i in (1, 2, 3))
        info = {k: np.concatenate([r[4][k] for r in results]) for k in results[0][4]}
        return obs, rew, term, trunc, info

    def _check_group(self, group):
        if not 0 <= group < self.groups:
            raise ValueError(f"group {group} out of range (groups={self.groups})")

    # -- API --
    def reset(self, seed=None):
        """-> obs, info. The first call returns the battles built at start-up (no rebuild) unless a seed is given;
        later calls rebuild every battle (seed: driver k gets seed + k*SEED_BLOCK + i; None: next seeds)."""
        if any(self._pending):
            raise RuntimeError("reset() with steps in flight; receive them first")
        if self._started or seed is not None:
            now = time.monotonic()
            for w in self._workers:
                self._busy[("reset", w.k)] = now + self.start_timeout
                w.send_reset(None if seed is None else seed + w.k * SEED_BLOCK)
            self._wait(lambda: not self._busy)
            self._last_info[:] = self._info
        self._started = True
        self._ep_ret[:] = 0
        self._ep_len[:] = 0
        info = {name: self._info[:, i].copy() for i, name in enumerate(INFO_FIELDS)}
        info["env_id"] = self._env_id.copy()
        return {"vec": self._vec.copy(), "grid": self._grid.copy()}, info

    def step_async(self, actions, group=0, repeat=0):
        """Send one action row per battle of the group (float32 [group_size, 10]); repeat <= 0: the driver default."""
        self._check_group(group)
        if self._pending[group]:
            raise RuntimeError(f"group {group} already has a step in flight")
        a = np.ascontiguousarray(actions, np.float32)
        if a.shape != (self.group_size, self.act_dim):
            raise ValueError(f"actions must have shape ({self.group_size}, {self.act_dim}), got {a.shape}")
        self._started = True
        self._pending[group] = True
        self._acts[group] = a
        self._busy[group] = time.monotonic() + self.timeout
        self._workers[group // self.pipeline].send_step(group, a, repeat, (group % self.pipeline) * self.group_size)

    def step_wait(self, group=0):
        """-> obs, reward, terminated, truncated, info for the group's rows (see the module doc)."""
        self._check_group(group)
        if not self._pending[group]:
            raise RuntimeError(f"group {group} has no step in flight")
        self._wait(lambda: group not in self._busy)
        return self._out([self._finish(group)])

    def send(self, actions, env_id=None, repeat=0):
        """Actions for whole groups: env_id = the rows they are for, in order (info["env_id"] of recv()), covering
        complete groups; None = all rows."""
        a = np.ascontiguousarray(actions, np.float32)
        if env_id is None:
            env_id = self._env_id
        env_id = np.asarray(env_id)
        if len(env_id) != len(a) or len(a) % self.group_size:
            raise ValueError(f"send: {len(a)} action rows for {len(env_id)} env ids; whole groups of "
                             f"{self.group_size} rows needed")
        for j in range(0, len(a), self.group_size):
            g = int(env_id[j]) // self.group_size
            if env_id[j] != g * self.group_size or env_id[j + self.group_size - 1] != (g + 1) * self.group_size - 1:
                raise ValueError("send: env_id must list whole groups in row order (as recv() returned them)")
            self.step_async(a[j:j + self.group_size], g, repeat)

    def recv(self, min_groups=1):
        """Wait until at least min_groups of the groups in flight have finished their step (all, if fewer are in
        flight) and return every finished group's rows: obs, reward, terminated, truncated, info (info["env_id"])."""
        pending = [g for g in range(self.groups) if self._pending[g]]
        if not pending:
            raise RuntimeError("recv() with nothing in flight; send() first")
        need = min(min_groups, len(pending))
        while True:
            while self._take(block=False):
                pass
            ready = [g for g in pending if g not in self._busy]
            if len(ready) >= need:
                break
            self._take(block=True)
        return self._out([self._finish(g) for g in ready])

    def step(self, actions, repeat=0):
        """All rows in lockstep: actions [num_envs, 10] -> results for all rows (slower than send/recv)."""
        self.send(actions, None, repeat)
        return self.recv(min_groups=self.groups)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for w in self._workers:
            if w.thread.is_alive():
                w.send_quit()
        for w in self._workers:
            w.thread.join(timeout=15)
            if w.thread.is_alive() and w.d is not None:
                w.d.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:                                          # noqa: BLE001
            pass


def random_actions(rng, rows, act_dim=len(ACT_FIELDS)):
    a = np.zeros((rows, act_dim), np.float32)
    a[:, :2] = rng.uniform(-1, 1, (rows, 2))
    a[:, 2] = rng.random(rows) < 0.3
    a[:, 5] = 1
    return a


def bench(mode, drivers, n, repeat, pipeline, seconds, policy_ms, min_groups=1, warmup=0.0, cpus=None, **battle):
    """random actions + a fake policy that takes policy_ms per call. mode: sync (step), async (send/recv)"""
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    with BrawlVecEnv(drivers=drivers, n=n, repeat=repeat, pipeline=pipeline, cpus=cpus, **battle) as env:
        startup = time.perf_counter() - t0

        def policy(rows):            # sleeps like torch inference (which releases the GIL)
            a = random_actions(rng, rows)
            if policy_ms:
                time.sleep(policy_ms / 1e3)
            return a
        env.reset()
        if warmup and mode == "async":    # battles all start together; play until games end at a steady rate
            env.send(policy(env.num_envs))
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < warmup:
                _, _, _, _, info = env.recv(min_groups)
                env.send(policy(len(info["env_id"])), info["env_id"])
            env.recv(env.groups)
        steps = episodes = calls = 0
        t0 = time.perf_counter()
        if mode == "sync":
            while time.perf_counter() - t0 < seconds:
                _, _, term, trunc, _ = env.step(policy(env.num_envs))
                steps += env.num_envs
                episodes += int((term | trunc).sum())
                calls += 1
        else:
            env.send(policy(env.num_envs))
            while time.perf_counter() - t0 < seconds:
                _, _, term, trunc, info = env.recv(min_groups)
                rows = len(term)
                steps += rows
                episodes += int((term | trunc).sum())
                calls += 1
                env.send(policy(rows), info["env_id"])
            env.recv(env.groups)
        dt = time.perf_counter() - t0
    print(f"{mode:5s} drivers={drivers} n={n} repeat={repeat} pipeline={pipeline} policy={policy_ms}ms: "
          f"{steps / dt:7.0f} env steps/s {steps * repeat / dt:7.0f} ticks/s | {steps / calls:5.0f} rows per policy "
          f"call | {episodes / dt:5.1f} games/s ({episodes} in {dt:.0f}s), startup {startup:.1f}s, crashes "
          f"{len(env.crashes)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--mode", nargs="+", default=["async"], choices=["sync", "async"])
    ap.add_argument("--drivers", type=int, default=8)
    ap.add_argument("--n", type=int, nargs="+", default=[96])
    ap.add_argument("--repeat", type=int, nargs="+", default=[4])
    ap.add_argument("--pipeline", type=int, nargs="+", default=[2])
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--policy-ms", type=float, nargs="+", default=[0])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--warmup", type=float, default=0, help="seconds played before measuring (games/s needs ~60)")
    ap.add_argument("--pin", action="store_true", help="pin driver k to tablet core k %% 8")
    ap.add_argument("kv", nargs="*", help="battle keys, e.g. hero=ShotgunGirl")
    a = ap.parse_args()
    battle = dict(s.split("=", 1) for s in a.kv)
    if not a.bench:
        ap.print_help()
        return
    for mode in a.mode:
        for n in a.n:
            for r in a.repeat:
                for p in a.policy_ms:
                    for pl in a.pipeline:
                        cpus = [[k % 8] for k in range(a.drivers)] if a.pin else None
                        bench(mode, a.drivers, n, r, pl, a.seconds, p, a.min_groups, a.warmup, cpus, **battle)


if __name__ == "__main__":
    main()
