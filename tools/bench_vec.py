#!/usr/bin/env python3
"""Throughput of the driver's vec mode with a random policy.

  python3 tools/bench_vec.py                         # n=1,8,32 x repeat 4, one driver
  python3 tools/bench_vec.py --n 32 --repeat 4 8 --seconds 20
  python3 tools/bench_vec.py --n 16 --drivers 4      # several drivers stepped in lockstep (wait for all each batch)
  python3 tools/bench_vec.py --n 16 --drivers 8 --async   # each driver gets its next actions as soon as it replies
  python3 tools/bench_vec.py --n 32 --drivers 8 --threads # one host thread per driver (fastest: replies of several
                                                          # drivers are received concurrently)
  python3 tools/bench_vec.py --n 16 --cpus 7         # pin the driver(s) to tablet cores (one per driver, cycled)
Reports env steps/s (agent decisions), ticks/s, and per batch: round trip vs. time the driver was busy simulating.
"""
import argparse
import pathlib
import selectors
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import VecDriver  # noqa: E402


def random_actions(rng, n, act_dim):
    a = np.zeros((n, act_dim), np.float32)
    a[:, :2] = rng.uniform(-1, 1, (n, 2))
    a[:, 2] = rng.random(n) < 0.3
    a[:, 5] = 1
    return a


def run(n, repeat, drivers, seconds, cpus, extra, asynchronous, threads=False):
    rng = np.random.default_rng(0)
    t_start = time.perf_counter()
    ds = [VecDriver(n=n, repeat=repeat, seed=1 + k * n, cpus=None if cpus is None else [cpus[k % len(cpus)]],
                    connect=False, **extra) for k in range(drivers)]
    for d in ds:                                   # the drivers load their tables in parallel
        d.connect()
    startup = time.perf_counter() - t_start
    try:
        batches, busy, resets, episodes = 0, [], 0, 0

        def done(d):
            nonlocal resets, episodes
            _, _, term, trunc, _ = d.step_wait()
            busy.append(d.busy_us)
            resets += d.reset_us
            episodes += int((term | trunc).sum())

        t0 = time.perf_counter()
        if threads:
            lock, stop = threading.Lock(), t0 + seconds

            def loop(k):
                nonlocal batches, resets, episodes
                r = np.random.default_rng(k)
                while time.perf_counter() < stop:
                    ds[k].step_async(random_actions(r, n, ds[k].act_dim))
                    _, _, term, trunc, _ = ds[k].step_wait()
                    with lock:
                        busy.append(ds[k].busy_us)
                        resets += ds[k].reset_us
                        episodes += int((term | trunc).sum())
                        batches += 1

            ts = [threading.Thread(target=loop, args=(k,)) for k in range(drivers)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            batches /= drivers
        elif asynchronous:
            sel = selectors.DefaultSelector()
            for d in ds:
                sel.register(d.proc.stdout, selectors.EVENT_READ, d)
                d.step_async(random_actions(rng, d.n, d.act_dim))
            while time.perf_counter() - t0 < seconds:
                for key, _ in sel.select():
                    done(key.data)
                    batches += 1
                    key.data.step_async(random_actions(rng, n, key.data.act_dim))
            batches /= drivers
        else:
            while time.perf_counter() - t0 < seconds:
                for d in ds:
                    d.step_async(random_actions(rng, d.n, d.act_dim))
                for d in ds:
                    done(d)
                batches += 1
        dt = time.perf_counter() - t0
    finally:
        for d in ds:
            d.close()
    steps = batches * n * drivers
    mode = " threads" if threads else " async" if asynchronous else ""
    print(f"n={n:3d} repeat={repeat} drivers={drivers}{mode}: {steps / dt:7.0f} env steps/s"
          f" {steps * repeat / dt:7.0f} ticks/s | batch {1e3 * dt / batches:6.2f} ms, driver busy {np.mean(busy) / 1e3:6.2f} ms"
          f" ({np.mean(busy) / (n * repeat) / 1e3:.3f} ms/tick, resets {resets / 1e3 / dt / drivers / 10:.1f}%)"
          f" | {episodes} episodes, startup {startup:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--repeat", type=int, nargs="+", default=[4])
    ap.add_argument("--drivers", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--cpus", type=int, nargs="+", default=None)
    ap.add_argument("--async", dest="asynchronous", action="store_true")
    ap.add_argument("--threads", action="store_true", help="one host thread per driver")
    ap.add_argument("kv", nargs="*", help="driver keys, e.g. loc=Gemgrab1")
    a = ap.parse_args()
    extra = dict(s.split("=", 1) for s in a.kv)
    for n in a.n:
        for r in a.repeat:
            run(n, r, a.drivers, a.seconds, a.cpus, extra, a.asynchronous, a.threads)


if __name__ == "__main__":
    main()
