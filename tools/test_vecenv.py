#!/usr/bin/env python3
"""Checks for BrawlVecEnv (tools/bsvecenv.py) on the tablet, ~3 min:

  exact      2 drivers in the env == 2 plain VecDrivers with the same seeds and actions (obs, grid, pieces, info, reward),
             lockstep step() and single groups stepped out of order, through several auto-resets
  async      send/recv: groups advance at their own pace, every row still sees exactly the plain driver's battle
  episodes   episode_return / episode_length == the sums of reward / steps of each finished battle
  crash      kill -9 one driver mid-run: its rows come back truncated + crashed with reward 0, it restarts with seeds
             above all it used, the other driver keeps matching its reference exactly
  hang       SIGSTOP one driver (timeout=5): killed, restarted, same as crash
  held half  a crash while the host holds one half of the driver: that half reports it on its next step
  watch      EnvRecorder episodes re-simulate exactly in serve mode (also checks range steps == serve mode)
  limit      more restarts than max_restarts raise RuntimeError
  leak       no driver process left on the tablet after close()
"""
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import INFO_FIELDS, SERIAL, TRUNCATED, VecDriver  # noqa: E402
from bsvecenv import SEED_BLOCK, BrawlVecEnv, random_actions  # noqa: E402

N, SEED, REPEAT = 8, 50_000_000, 8
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail and not ok else ''}")
    if not ok:
        FAILS.append(name)


def drivers_on_tablet():
    out = subprocess.run(["adb", "-s", SERIAL, "shell", "ps -A -o NAME"], capture_output=True, text=True).stdout
    return sum(line.strip() == "driver" for line in out.splitlines())


def signal(env, k, sig):
    pid = env._workers[k].d.remote_pid
    subprocess.run(["adb", "-s", SERIAL, "shell", f"su -c 'kill -{sig} {pid}'"], capture_output=True, timeout=15)


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def test_exact(groups, steps=600):
    print(f"exact ({'lockstep step()' if groups == 1 else 'single groups out of order'}, {steps} steps of {REPEAT} ticks)")
    rng = np.random.default_rng(1)
    refs = [VecDriver(n=N, repeat=REPEAT, seed=SEED + k * SEED_BLOCK, connect=False) for k in range(2)]
    for d in refs:
        d.connect()
    bad, ends = [], 0
    with BrawlVecEnv(drivers=2, n=N, repeat=REPEAT, seed=SEED) as env:
        obs, info = env.reset()
        ref_vec = np.concatenate([d.obs["vec"] for d in refs])
        if not same(obs["vec"], ref_vec):
            bad.append("reset obs")
        for t in range(steps):
            act = random_actions(rng, 2 * N)
            if groups == 1:
                obs, rew, term, trunc, info = env.step(act)
            else:   # groups (driver halves) sent in reverse order, waited for in a shuffled order
                G = env.groups
                for g in reversed(range(G)):
                    env.step_async(act[env.group_slice(g)], g)
                rs = {g: env.step_wait(g) for g in rng.permutation(G).tolist()}
                rs = [rs[g] for g in range(G)]
                obs = {k: np.concatenate([r[0][k] for r in rs]) for k in rs[0][0]}
                rew, term, trunc = (np.concatenate([r[i] for r in rs]) for i in (1, 2, 3))
                info = {k: np.concatenate([r[4][k] for r in rs]) for k in ("parts", *INFO_FIELDS)}
            out = [d.step(act[k * N:(k + 1) * N]) for k, d in enumerate(refs)]
            want = {"vec": np.concatenate([o[0]["vec"] for o in out]), "grid": np.concatenate([o[0]["grid"] for o in out]),
                    "rew": np.concatenate([o[1] for o in out]), "info": np.concatenate([o[4] for o in out]),
                    "parts": np.concatenate([d.parts for d in refs])}
            got_info = np.stack([info[f] for f in INFO_FIELDS], 1)
            for name, a, b in (("vec", obs["vec"], want["vec"]), ("grid", obs["grid"], want["grid"]),
                               ("reward", rew, want["rew"]), ("info", got_info, want["info"]),
                               ("parts", info["parts"], want["parts"])):
                if not same(a, b):
                    bad.append(f"step {t} {name}")
            ends += int((term | trunc).sum())
            if len(bad) > 5:
                break
    for d in refs:
        d.close()
    check(f"identical to plain drivers, {ends} battles ended", not bad and ends > 0, ", ".join(bad[:5]))


def act_for(env_id, t):
    """actions that depend only on (battle row, its step count), so any stepping order gives the same battles"""
    return np.stack([random_actions(np.random.default_rng((int(i), int(t[i]))), 1)[0] for i in env_id])


def test_async(steps=300):
    print(f"async (send/recv, 4 drivers x {N}, {steps} steps per driver)")
    D = 4
    refs = [VecDriver(n=N, repeat=REPEAT, seed=SEED + 300 + k * SEED_BLOCK, connect=False) for k in range(D)]
    for d in refs:
        d.connect()
    bad, t, uneven = [], np.zeros(D * N, int), 0
    with BrawlVecEnv(drivers=D, n=N, repeat=REPEAT, seed=SEED + 300) as env:
        obs, info = env.reset()
        env.send(act_for(info["env_id"], t), info["env_id"])
        got = {}
        while True:
            obs, rew, term, trunc, info = env.recv()
            ids = info["env_id"]
            uneven += len(ids) < D * N
            for j, i in enumerate(ids):
                got[(int(i), int(t[i]))] = (obs["vec"][j].copy(), float(rew[j]), int(info["flags"][j]))
            t[ids] += 1
            if t.min() >= steps:
                if any(env._pending):
                    env.recv(env.groups)
                break
            ids = ids[t[ids] < steps + 5]
            if len(ids):
                env.send(act_for(ids, t), ids)
    for k, d in enumerate(refs):
        rows = np.arange(k * N, (k + 1) * N)
        tt = np.zeros(D * N, int)
        for s in range(steps):
            o, rew, term, trunc, inf = d.step(act_for(rows, tt))
            for j, i in enumerate(rows):
                g = got[(int(i), s)]
                if not (np.array_equal(g[0], o["vec"][j]) and g[1] == float(rew[j]) and g[2] == int(inf[j, 0])):
                    bad.append(f"row {i} step {s}")
            tt[rows] += 1
        d.close()
    check(f"every row identical to its plain driver ({uneven} recv() calls returned only some groups)",
          not bad and uneven > 0, ", ".join(bad[:5]))


def test_episodes(steps=700):
    print("episodes")
    rng = np.random.default_rng(2)
    with BrawlVecEnv(drivers=2, n=N, repeat=REPEAT, seed=SEED + 7) as env:
        env.reset()
        acc, ln, bad, done = np.zeros(2 * N), np.zeros(2 * N, int), 0, 0
        env.set_weights({"win": 2.0, "dmg_dealt": 0.01})
        for _ in range(steps):
            _, rew, term, trunc, info = env.step(random_actions(rng, 2 * N))
            if not np.allclose(rew, info["parts"] @ env._w, atol=1e-6):
                bad += 1
            acc += rew
            ln += 1
            f = info["final"]
            if not (np.allclose(info["episode_return"][f], acc[f], atol=1e-4) and np.array_equal(info["episode_length"][f], ln[f])
                    and (info["episode_return"][~f] == 0).all()):
                bad += 1
            done += int(f.sum())
            acc[f], ln[f] = 0, 0
    check(f"episode stats and reweighted reward ({done} episodes)", bad == 0 and done > 0, f"{bad} bad steps")


def test_failure(kind):
    print(f"{kind} (driver 1 {'killed' if kind == 'crash' else 'stopped'} mid-run)")
    rng = np.random.default_rng(3)
    ref = VecDriver(n=N, repeat=REPEAT, seed=SEED + 100)
    with BrawlVecEnv(drivers=2, n=N, repeat=REPEAT, seed=SEED + 100, timeout=5) as env:
        env.reset()
        used, bad, crash_rows, after = set(), [], None, 0
        for t in range(120):
            if t == 40:
                old_pid = env._workers[1].d.remote_pid
                signal(env, 1, 9 if kind == "crash" else "STOP")
            act = random_actions(rng, 2 * N)
            t0 = time.perf_counter()
            obs, rew, term, trunc, info = env.step(act)
            dt = time.perf_counter() - t0
            o = ref.step(act[:N])
            if not (same(obs["vec"][:N], o[0]["vec"]) and same(rew[:N], o[1])):
                bad.append(f"driver 0 diverged at step {t}")
            if info["crashed"].any():
                crash_rows = (t, np.flatnonzero(info["crashed"]).tolist(), trunc.copy(), rew.copy(), dt)
            if crash_rows is None:
                used.update(info["seed"][N:].tolist())
            elif t > crash_rows[0]:
                after += 1
                if info["crashed"].any():
                    bad.append(f"crashed again at step {t}")
                if t == crash_rows[0] + 1:
                    new_seeds = info["seed"][N:]
        ref.close()
        check("crash rows = driver 1's battles", crash_rows is not None and crash_rows[1] == list(range(N, 2 * N)),
              str(crash_rows and crash_rows[1]))
        if crash_rows:
            t, _, trunc, rew, dt = crash_rows
            check("crash rows truncated, reward 0", trunc[N:].all() and not rew[N:].any())
            check("restarted driver is a new process", env._workers[1].d.remote_pid != old_pid)
            check(f"new seeds above all used ({min(new_seeds)} > {max(used)})", min(new_seeds) > max(used))
            check("one crash recorded", len(env.crashes) == 1, str(env.crashes))
            print(f"    recovery step took {dt:.1f}s; {after} normal steps after it")
        check("driver 0 unaffected (matches its reference)", not bad, ", ".join(bad[:3]))


def test_held_half():
    print("crash while the host holds one half of the driver")
    rng = np.random.default_rng(5)
    with BrawlVecEnv(drivers=1, n=N, repeat=REPEAT, seed=SEED + 400) as env:
        env.reset()
        h = N // 2
        for _ in range(20):
            env.step(random_actions(rng, N))
        env.step_async(random_actions(rng, h), 0)
        env.step_wait(0)                                  # half 0 is now with the host
        seeds0 = env._last_info[:h, 4].copy()
        signal(env, 0, 9)
        time.sleep(1)
        env.step_async(random_actions(rng, h), 1)
        _, rew1, _, trunc1, info1 = env.step_wait(1)      # sent to the dead driver
        env.step_async(random_actions(rng, h), 0)
        _, rew0, _, trunc0, info0 = env.step_wait(0)      # held: its battles were lost too
        env.step_async(random_actions(rng, h), 0)
        _, _, _, trunc0b, info0b = env.step_wait(0)
        check("in-flight half: truncated + crashed", trunc1.all() and info1["crashed"].all() and not rew1.any())
        check("held half: truncated + crashed on its next step, old seeds reported",
              trunc0.all() and info0["crashed"].all() and not rew0.any() and np.array_equal(info0["seed"], seeds0))
        check("held half: normal again after that", not trunc0b.any() and not info0b["crashed"].any()
              and info0b["seed"].min() > seeds0.max())


def test_watch():
    print("watch (EnvRecorder on 2 battles in different drivers/halves, games re-simulated in serve mode)")
    import contextlib
    import io
    from episodes import EnvRecorder, rerender
    rng = np.random.default_rng(6)
    with BrawlVecEnv(drivers=2, n=N, repeat=4, seed=SEED + 500) as env:
        rec = EnvRecorder(env, env_ids=[0, N + N // 2 + 1], extras={"value": lambda ids: ids * 0.5})
        obs, info = env.reset()
        env.send(random_actions(rng, 2 * N))
        while len(rec.episodes) < 2:
            obs, rew, term, trunc, info = env.recv()
            env.send(random_actions(rng, len(term)), info["env_id"])
        env.recv(env.groups)
    ok = True
    for ep in rec.episodes[:2]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rerender(ep, check=True)
        ok &= "diverged" not in err.getvalue() and len(ep.columns["value"]) == len(ep.actions)
    check(f"recorded games re-simulate exactly (seeds {[e.seed for e in rec.episodes[:2]]})", ok)


def test_limit():
    print("limit (max_restarts=1, driver 0 killed twice)")
    rng = np.random.default_rng(4)
    raised = False
    with BrawlVecEnv(drivers=1, n=4, repeat=REPEAT, seed=SEED + 200, max_restarts=1) as env:
        env.reset()
        try:
            for t in range(30):
                if t in (5, 15):
                    signal(env, 0, 9)
                env.step(random_actions(rng, 4))
        except RuntimeError as e:
            raised = "restarts within" in str(e)
    check("raises after too many restarts", raised)


def main():
    t0 = time.perf_counter()
    before = drivers_on_tablet()
    if before:
        print(f"warning: {before} driver processes already running on the tablet")
    test_exact(1)
    test_exact(2)
    test_async()
    test_episodes()
    test_failure("crash")
    test_failure("hang")
    test_held_half()
    test_watch()
    test_limit()
    time.sleep(2)
    print("leak")
    check("no driver process left on the tablet", drivers_on_tablet() == before, f"{drivers_on_tablet()} running")
    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)} ({time.perf_counter() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
