#!/usr/bin/env python3
"""Soak test for BrawlVecEnv: the training configuration (8 drivers x 96, pipelined) with random actions and an
EnvRecorder attached, for a long time. Every `--every` seconds it prints decisions/s, games/s, crashes, the drivers'
memory (RSS on the tablet) and the tablet's hottest temperature, then judges the trends:
  memory   RSS of the last sample vs the first after warm-up (leak if it keeps growing)
  speed    decisions/s of the last samples vs the first (thermal throttling)
  crashes  restarts during the run

  tools/soak_vecenv.py --minutes 20
"""
import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import SERIAL  # noqa: E402
from bsvecenv import BrawlVecEnv, random_actions  # noqa: E402
from episodes import EnvRecorder  # noqa: E402


def tablet_stats(serial=SERIAL):
    """-> (total driver RSS MB, max per-driver RSS MB, hottest thermal zone in C)"""
    out = subprocess.run(["adb", "-s", serial, "shell", "ps -A -o RSS,NAME; cat /sys/class/thermal/thermal_zone*/temp"],
                         capture_output=True, text=True, timeout=30).stdout.split("\n")
    rss = [int(line.split()[0]) / 1024 for line in out if line.strip().endswith(" driver")]
    temps = [int(line) for line in out if line.strip().lstrip("-").isdigit()]
    hot = max((t / 1000 if t > 1000 else t for t in temps), default=float("nan"))
    return sum(rss), max(rss, default=0), hot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--drivers", type=int, default=8)
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    with BrawlVecEnv(drivers=a.drivers, n=a.n) as env:
        rec = EnvRecorder(env, env_ids=[0, env.num_envs - 1])
        obs, info = env.reset()
        env.send(random_actions(rng, env.num_envs))
        t_end = time.time() + 60 * a.minutes
        print(f"soak: {a.drivers} drivers x {a.n} battles for {a.minutes:.0f} min")
        print(f"{'min':>5} {'dec/s':>7} {'games/s':>7} {'crashes':>7} {'RSS MB':>7} {'max/drv':>7} {'temp C':>6}")
        t0 = time.time()
        while time.time() < t_end:
            steps = games = 0
            ts = time.time()
            while time.time() - ts < a.every:
                obs, rew, term, trunc, info = env.recv()
                steps += len(term)
                games += int((term | trunc).sum())
                env.send(random_actions(rng, len(term)), info["env_id"])
            dt = time.time() - ts
            rss, rmax, hot = tablet_stats()
            rows.append(((time.time() - t0) / 60, steps / dt, games / dt, len(env.crashes), rss, rmax, hot))
            print("{:5.1f} {:7.0f} {:7.1f} {:7d} {:7.0f} {:7.1f} {:6.1f}".format(*rows[-1]), flush=True)
        env.recv(env.groups)
        rec.close()
        recorded = len(rec.episodes)
    r = np.array(rows)
    k = max(1, len(r) // 4)
    first, last = r[1:1 + k] if len(r) > k + 1 else r[:k], r[-k:]
    growth = last[:, 4].mean() - first[:, 4].mean()
    speed = last[:, 1].mean() / first[:, 1].mean()
    print(f"memory: {first[:, 4].mean():.0f} -> {last[:, 4].mean():.0f} MB total ({growth:+.0f} MB)  "
          f"speed: {first[:, 1].mean():.0f} -> {last[:, 1].mean():.0f} decisions/s (x{speed:.2f})  "
          f"crashes: {int(r[-1, 3])}  recorded episodes: {recorded}")
    ok = growth < 50 and speed > 0.85 and r[-1, 3] == 0 and recorded > 0
    print("SOAK PASS" if ok else "SOAK CHECK: see the numbers above")


if __name__ == "__main__":
    main()
