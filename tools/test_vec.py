#!/usr/bin/env python3
"""Checks for the driver's vec mode (tools/bsvec.py). Run with the tablet attached: python3 tools/test_vec.py

1. same as serve: a vec battle and a serve-mode battle (bsenv.BrawlEnv, unmodified tick incl. vision encoding) with the
   same seed and the same random actions: every observation field (scores, own state, heroes, velocities, gems,
   projectiles, area effects, tile grid) recomputed from the serve frames matches, every step, until the same game over;
   hidden enemies are plausible (in a bush or far from our team)
   reward pieces: vec == serve every step, and each piece agrees with the frames (score/gem changes, own deaths, hp
   lost with the low-hp weighting, seconds below 10% hp, win at game over; kills/damage only when an enemy brawler
   died / an enemy's hp changed), episode sums of gem_diff/countdown = final difference / +-1 when a countdown ended the game
2. deterministic: two drivers with the same seeds and actions produce identical observations
3. isolated: battle i of an n=4 driver == the same seed in an n=1 driver, across an automatic reset
4. auto-reset: done rows carry the finished battle's seed/score/result, the next obs is a fresh battle (seed + n)
5. memory: RSS of a driver after ~100 episodes vs. right after start
6. errors: an unknown map name fails with the driver's message
"""
import collections
import csv
import pathlib
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsenv import Action, BrawlEnv  # noqa: E402
from bsvec import DEV_DIR, INFO_FIELDS, REWARD_PARTS, SERIAL, VecDriver  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def random_actions(rng, n):
    a = np.zeros((n, 10), np.float32)
    a[:, :2] = rng.uniform(-1, 1, (n, 2))
    a[:, 2] = rng.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15])
    a[:, 3:5] = rng.uniform(-1, 1, (n, 2))
    a[:, 5] = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
    a[:, 6:8] = rng.uniform(-1, 1, (n, 2))
    return a


TILE_CSV = ROOT / "scratch/tiles_v36.csv"


def tile_flag_table():
    """tile name -> grid flags, from the tiles.csv the driver loads (pulled from the tablet once)"""
    if not TILE_CSV.exists():
        subprocess.run(["adb", "-s", SERIAL, "pull", f"{DEV_DIR}/assets/csv_logic/tiles.csv", str(TILE_CSV)],
                       check=True, capture_output=True)
    rows = list(csv.reader(TILE_CSV.open()))
    col = {h: i for i, h in enumerate(rows[0])}
    out = {}
    for r in rows[2:]:
        get = lambda c: r[col[c]] if c in col and col[c] < len(r) else ""
        f = (1 if get("BlocksMovement") == "true" else 0) | (2 if get("BlocksProjectiles") == "true" else 0) | \
            (4 if get("HidesHero") == "true" else 0) | (8 if get("IsDestructible") == "true" else 0) | \
            (32 if int(get("Damage") or 0) or int(get("SpeedChange") or 0) else 0)
        out[r[col["Name"]]] = f
    return out


class ObsCheck:
    """Recomputes the vec observation from serve-mode frames (all objects, full tile map) and lists differences.
    Fog of war cannot be recomputed from frames; it is checked for plausibility: an enemy hidden while standing in the
    open within 2000 units of a living teammate is an error, hidden-in-bush / hidden-far / revealed counts are kept."""

    def __init__(self, d, meta, tiles):
        self.d, self.meta, self.tiles = d, meta, tiles
        self.W, self.H, self.w = meta["w"] * 300, meta["h"] * 300, meta["w"]
        self.team, self.name, self.static = {}, {}, {}
        self.last_own = self.prev_tick = self.bt = None
        self.prev, self.last_pos, self.vis_pos = {}, {}, {}
        self.resp = {}                                    # slot -> respawn value at the previous step
        self.stats = collections.Counter()

    def respawn(self, err, tag, alive, value):
        """respawn timer: ~1 right after death, falls every step, ~0 on the step before the hero is back"""
        prev = self.resp.get(tag)
        if not alive:
            if prev is None and value < 0.85:
                err.append(f"{tag}.respawn {value:.3f} on the first dead step")
            if prev is not None and value > prev + 1e-6:
                err.append(f"{tag}.respawn rose {prev:.3f} -> {value:.3f}")
            self.resp[tag] = float(value)
        elif prev is not None:
            self.stats["respawns seen"] += 1
            self.stats["max respawn value before respawning"] = max(self.stats["max respawn value before respawning"],
                                                                    round(prev, 3))
            if prev > 0.1:
                err.append(f"{tag} respawned while its timer said {prev:.3f}")
            self.resp.pop(tag)

    def flags_at(self, grid, tx, ty):
        if tx < 0 or ty < 0 or tx >= self.w or ty >= self.meta["h"]:
            return 16 | 1 | 2
        return self.tiles.get(grid[ty * self.w + tx], 0)

    def __call__(self, vec, grid, ref):
        err, d, tick = [], self.d, ref.tick
        near = lambda a, b, tol=1e-3: abs(float(a) - float(b)) <= tol
        expect = lambda what, got, want, tol=1e-3: near(got, want, tol) or err.append(f"{what}: {got:.5g} != {want:.5g}")
        objs = ref.objects
        for o in objs:                                    # first frame: every player's hero is alive
            if o.type == "brawler" and o.player >= 0 and o.player not in self.name:
                self.name[o.player], self.team[o.player] = o.name, o.team
        heroes = {o.player: o for o in objs if o.type == "brawler" and o.player >= 0 and o.hp > 0
                  and self.name.get(o.player) == o.name}
        me = heroes.get(0) if ref.alive else None
        if me:
            self.last_own = (me.x, me.y)
        ox, oy = self.last_own
        dt = tick - self.prev_tick if self.prev_tick is not None else 1
        for p, h in heroes.items():
            self.last_pos[p] = (h.x, h.y)

        g = {b: {f: a for f, a in d.view(vec, b).items()} for b in d.blocks}
        # global
        G = {f: v[0] for f, v in g["global"].items()}
        expect("own_score", G["own_score"], ref.score[0] / 10)
        expect("enemy_score", G["enemy_score"], ref.score[1] / 10)
        if self.bt is None and G["time"] > 0:
            self.bt = tick / G["time"]
        if self.bt:
            expect("time*battle_ticks", G["time"] * self.bt, tick, 0.6)
        mine = next((o for o in objs if o.name == "OrbSpawner"), None)
        expect("mine", G["mine"], mine is not None)
        if mine:
            expect("mine_dx", G["mine_dx"] * 3000, mine.x - ox, 0.5)
            expect("mine_dy", G["mine_dy"] * 3000, mine.y - oy, 0.5)
        if G["own_countdown"] > 0 and not (ref.score[0] > ref.score[1] and ref.score[0] >= 10):
            err.append(f"own countdown {G['own_countdown']} with score {ref.score}")
        if G["enemy_countdown"] > 0 and not (ref.score[1] > ref.score[0] and ref.score[1] >= 10):
            err.append(f"enemy countdown {G['enemy_countdown']} with score {ref.score}")
        for f, team in (("own_countdown", 0), ("enemy_countdown", 1)):   # vs the frame's countdown (ticks left)
            want = ref.countdown if ref.countdown_team == team else 0
            if (G[f] > 0) != (want > 0):
                err.append(f"{f} {G[f]:.3f} but frame countdown {ref.countdown} for team {ref.countdown_team}")
            elif want:
                self.stats["countdown steps"] += 1
                expect(f"{f} length", want / G[f], self.static.setdefault("cd_len", want / float(G[f])), 0.6)
        # self
        S = {f: v[0] for f, v in g["self"].items()}
        expect("self.alive", S["alive"], me is not None)
        expect("self.x", S["x"] * self.W, ox, 0.5)
        expect("self.y", S["y"] * self.H, oy, 0.5)
        if me:
            pv = self.prev.get(0)
            expect("self.vx", S["vx"], (me.x - pv[0]) / dt / 40 if pv else 0)
            expect("self.vy", S["vy"], (me.y - pv[1]) / dt / 40 if pv else 0)
            expect("self.hp", S["hp"], me.hp / me.max_hp)
            expect("self.max_hp", S["max_hp"], me.max_hp / 5000)
            expect("self.ammo", S["ammo"], ref.ammo / 3)
            expect("self.ammo_max", S["ammo_max"], ref.ammo_max / 3)
            expect("self.range", S["range"], self.meta["range"][0] / 3000)
            expect("self.super_range", S["super_range"], self.meta["range"][1] / 3000)
            expect("self.respawn", S["respawn"], 0)
            expect("self.gems", S["gems"] * 10, me.gems)
        else:
            self.stats["own dead steps"] += 1
        self.respawn(err, "self", me is not None, S["respawn"])
        if ref.super_needed > 0:
            expect("self.super", S["super"], min(ref.super_charge / ref.super_needed, 1))
            expect("self.super_ready", S["super_ready"], ref.super_charge >= ref.super_needed)
        # other players
        own_team = self.team[0]
        players = sorted(self.name)
        slots = [("ally", p) for p in players if p != 0 and self.team[p] == own_team][:2] + \
                [("enemy", p) for p in players if self.team[p] != own_team][:3]
        gems_team = S["gems"] * 10
        members = [h for p, h in heroes.items() if self.team[p] == own_team]
        for k, (kind, p) in enumerate(slots):
            k_in = k if kind == "ally" else k - 2
            R = {f: v[k_in] for f, v in g[kind].items()}
            h = heroes.get(p)
            tag = f"{kind}{k_in}(p{p})"
            expect(f"{tag}.present", R["present"], 1)
            expect(f"{tag}.alive", R["alive"], h is not None)
            if self.d.blocks[kind].fields.count("respawn"):
                self.respawn(err, tag, h is not None, R["respawn"])
            for f in ("hero", "range", "speed", "max_hp"):
                if R["alive"]:
                    want = self.static.setdefault((p, f), float(R[f]))
                    expect(f"{tag}.{f} constant", R[f], want)
            if kind == "ally":
                gems_team += R["gems"] * 10
                expect(f"{tag}.visible", R["visible"], h is not None)
            if R["visible"]:
                if h is None:
                    err.append(f"{tag} visible but dead")
                    continue
                expect(f"{tag}.dx", R["dx"] * 3000, h.x - ox, 0.5)
                expect(f"{tag}.dy", R["dy"] * 3000, h.y - oy, 0.5)
                expect(f"{tag}.hp", R["hp"], h.hp / h.max_hp)
                expect(f"{tag}.seen_age", R["seen_age"], 0)
                expect(f"{tag}.gems", R["gems"] * 10, h.gems)
                pv = self.prev.get(p)
                expect(f"{tag}.vx", R["vx"], (h.x - pv[0]) / dt / 40 if pv else 0)
                self.vis_pos[p] = (h.x, h.y)
                if kind == "enemy":
                    self.stats["enemy visible"] += 1
            else:
                last = self.vis_pos.get(p)
                if last:
                    expect(f"{tag}.dx (last seen)", R["dx"] * 3000, last[0] - ox, 0.5)
                    expect(f"{tag}.dy (last seen)", R["dy"] * 3000, last[1] - oy, 0.5)
                if kind == "enemy" and h is not None:
                    in_bush = self.flags_at(ref.grid, h.x // 300, h.y // 300) & 4
                    close = any(abs(h.x - m.x) <= 2000 and abs(h.y - m.y) <= 2000 for m in members)
                    self.stats["enemy hidden in bush" if in_bush else "enemy hidden far" if not close
                               else "enemy hidden IN THE OPEN NEARBY"] += 1
                    if close and not in_bush:
                        err.append(f"{tag} hidden in the open {h.x - ox},{h.y - oy} from us, near a teammate")
        if mine:   # gem grab: team score = gems carried by the team
            expect("self.gems + ally gems", gems_team, ref.score[0], 1e-3)
        # items, projectiles, area effects: every listed row is a real object of that kind, nearest first
        for block, kind in (("item", "item"), ("projectile", "projectile"), ("area", "area_effect")):
            B = g[block]
            cands = [o for o in objs if o.type == kind and o.name != "OrbSpawner"]
            rows = int(B["present"].sum())
            if rows > len(cands):
                err.append(f"{block}: {rows} rows but {len(cands)} objects")
            dists = []
            for i in range(rows):
                x, y = ox + B["dx"][i] * 3000, oy + B["dy"][i] * 3000
                m = [o for o in cands if abs(o.x - x) <= 0.5 and abs(o.y - y) <= 0.5]
                if not m:
                    err.append(f"{block}{i} at {x:.0f},{y:.0f}: no such {kind}")
                    continue
                if block == "item" and not any(near(B["z"][i], o.z / 1000) for o in m):
                    err.append(f"item{i}: z {B['z'][i]} vs {[o.z for o in m]}")
                if block != "item" and not any(near(B["enemy"][i], o.team != own_team) for o in m):
                    err.append(f"{block}{i}: enemy flag {B['enemy'][i]}")
                dists.append((x - ox) ** 2 + (y - oy) ** 2)
            if dists != sorted(dists):
                err.append(f"{block}: rows not sorted by distance")
            self.stats[f"{block} rows"] += rows
            self.stats[f"{block} objects"] += len(cands)
        # tile grid
        G2 = self.d.grid_size
        cx, cy = int(ox) // 300, int(oy) // 300
        want = np.array([[self.flags_at(ref.grid, cx - G2 // 2 + x, cy - G2 // 2 + y) for x in range(G2)]
                         for y in range(G2)], np.uint8)
        if not np.array_equal(grid, want):
            bad = np.argwhere(grid != want)[0]
            err.append(f"grid differs at {bad.tolist()}: {grid[tuple(bad)]} vs {want[tuple(bad)]}")
        self.prev = {p: (h.x, h.y) for p, h in heroes.items()}
        self.prev_tick = tick
        return err


class RewardCheck:
    """Recomputes the reward pieces from consecutive serve frames (bsenv.Obs) where the frames allow it."""

    def __init__(self, first):
        self.prev, self.sums, self.stats = first, collections.Counter(), collections.Counter()
        self.team = first.own.team

    @staticmethod
    def own_hp(o):          # fraction, -1 while dead
        me = o.own
        return me.hp / me.max_hp if o.alive and me and me.hp > 0 and me.max_hp > 0 else -1.0

    def enemies(self, o):   # gid -> enemy hero or summon (summons carry their owner's player index, so not by player)
        return {x.gid: x for x in o.objects if x.type == "brawler" and x.player >= 0 and x.team != self.team
                and x.hp > 0}

    def __call__(self, o):
        err, p, r = [], self.prev, o.reward_parts
        expect = lambda what, want, tol=2e-4: abs(r[what] - want) <= tol or err.append(f"{what}: {r[what]:.5g} != {want:.5g}")
        for k in REWARD_PARTS:
            self.sums[k] += r[k]
        expect("gem_diff", (o.score[0] - o.score[1]) - (p.score[0] - p.score[1]))
        gems = lambda x: x.own.gems if x.alive and x.own else 0
        expect("gems_carried", gems(o) - gems(p))
        expect("death", float(p.alive and not o.alive))
        h0, h1 = self.own_hp(p), max(0.0, self.own_hp(o))
        F = lambda h: 1.5 * h - 0.5 * h * h
        expect("dmg_taken", F(h0) - F(h1) if h0 >= 0 and h1 < h0 else 0)
        expect("low_hp", (o.tick - p.tick) / 20 if 0 <= self.own_hp(o) < 0.1 else 0)
        diff = o.score[0] - o.score[1]
        expect("win", float((diff > 0) - (diff < 0)) if o.done else 0)
        e0, e1 = self.enemies(p), self.enemies(o)
        died = [q for q in e0 if q not in e1]
        lost = sum(e0[q].hp - (e1[q].hp if q in e1 else 0) for q in e0 if q not in e1 or e1[q].hp < e0[q].hp)
        if r["kill"] > len(died):
            err.append(f"kill {r['kill']} but {len(died)} enemy deaths")
        changed = died or any(e1[q].hp != e0[q].hp for q in e0 if q in e1)   # hp can rise: regen in the same step
        if r["dmg_dealt"] > 0 and not changed:
            err.append(f"dmg_dealt {r['dmg_dealt']} but no enemy hp changed")
        self.stats["damage dealt"] += r["dmg_dealt"] * 1000
        self.stats["enemy hp lost"] += max(lost, 0)
        self.stats["kills"] += r["kill"]
        self.stats["enemy deaths"] += len(died)
        self.prev = o
        return err


def test_same_as_serve(seed=3, steps=1500, policy=None, **battle):
    """policy: act(obs, d) on the vec observation (eval.py baselines) instead of random actions"""
    rng = np.random.default_rng(seed)
    tiles = tile_flag_table()
    with VecDriver(n=1, seed=seed, **battle) as d, BrawlEnv(repeat=4) as env:
        ref = env.reset(seed=seed, **battle)
        chk = ObsCheck(d, env.meta, tiles)
        rchk = RewardCheck(ref)
        obs, first_err, n, ended = d.obs, None, 0, False
        rew_err = None
        for n in range(steps):
            err = chk(obs["vec"][0], obs["grid"][0], ref)
            if err and first_err is None:
                first_err = (n, ref.tick, err)
            a = random_actions(rng, 1) if policy is None else policy(obs, d)
            obs, _, term, trunc, info = d.step(a)
            ref = env.step(Action(move=tuple(a[0, :2]), attack=int(a[0, 2]), aim=tuple(a[0, 3:5]), super_=int(a[0, 5]),
                                  super_aim=tuple(a[0, 6:8])))
            e = rchk(ref)
            vp = dict(zip(d.reward_names, d.parts[0].tolist()))
            e += [f"vec {k} {vp[k]:.6g} != serve {ref.reward_parts[k]:.6g}" for k in REWARD_PARTS
                  if abs(vp[k] - ref.reward_parts[k]) > 1e-5 * max(1, abs(vp[k]))]
            if e and rew_err is None:
                rew_err = (n, ref.tick, e)
            if term[0] or trunc[0] or ref.done:
                same_end = bool(term[0]) == ref.done and (info[0, 2], info[0, 3]) == tuple(ref.score)
                check(f"same as serve (seed {seed}): game over on the same step with the same score", same_end,
                      f"step {n}: vec {dict(zip(INFO_FIELDS, info[0].tolist()))} serve done={ref.done} score={ref.score}")
                ended = True
                break
    where = battle.get("loc", "Gemgrab1") + (f", policy {policy.__name__}" if policy else "")
    st, sm = rchk.stats, rchk.sums
    check(f"reward pieces (seed {seed}, {where}): vec == serve and consistent with the frames every step",
          rew_err is None, (f"sums {', '.join(f'{k}={v:.3g}' for k, v in sm.items())}; kills {st['kills']:.0f}/"
                            f"{st['enemy deaths']} enemy deaths, dmg dealt {st['damage dealt']:.0f} of "
                            f"{st['enemy hp lost']:.0f} enemy hp lost") if rew_err is None else
          f"first error at step {rew_err[0]} (tick {rew_err[1]}): {rew_err[2][:6]}")
    if ended:
        diff = ref.score[0] - ref.score[1]
        gem_grab = env.meta["loc"].startswith("Gemgrab")
        cd_end = gem_grab and chk.bt and ref.tick < chk.bt - 1   # ended before the time limit: a countdown decided it
        ok = abs(sm["gem_diff"] - diff) < 1e-4 and (not cd_end or abs(sm["countdown"] - np.sign(diff)) < 1e-4)
        check(f"reward sums (seed {seed}, {where}): gem_diff = final difference, countdown = +-1 if a countdown ended it",
              ok, f"gem_diff {sm['gem_diff']:.4g} vs {diff}, countdown {sm['countdown']:.4g}, tick {ref.tick}, "
                  f"countdown ended it: {cd_end}")
    check(f"same as serve (seed {seed}, {where}): every observation field matches the serve frames", first_err is None,
          f"{n + 1} steps, {dict(chk.stats)}" if first_err is None else
          f"first error at step {first_err[0]} (tick {first_err[1]}): {first_err[2][:6]}")
    return ended


def run_actions(n, seed, steps, action_seed):
    rng = np.random.default_rng(action_seed)
    out = []
    with VecDriver(n=n, seed=seed) as d:
        out.append((d.obs, None))
        for _ in range(steps):
            obs, rew, term, trunc, info = d.step(random_actions(rng, n))
            out.append((obs, info))
    return out


def test_deterministic():
    a, b = run_actions(4, 11, 300, 5), run_actions(4, 11, 300, 5)
    same = all(np.array_equal(x[0]["vec"], y[0]["vec"]) and np.array_equal(x[0]["grid"], y[0]["grid"])
               for x, y in zip(a, b))
    check("deterministic: same seeds + actions -> identical observations (4 battles x 300 steps)", same)


def test_isolated_and_reset():
    rng = np.random.default_rng(7)
    steps = 800                      # Gem Grab ends after ~450-900 steps of 4 ticks: crosses at least one reset
    acts = [random_actions(rng, 4) for _ in range(steps)]
    with VecDriver(n=4, seed=21) as d4, VecDriver(n=1, seed=23) as d1:
        diff, ends, d1_ends = None, [], 0
        o4, o1 = d4.obs, d1.obs
        for t in range(steps):
            if diff is None and d1_ends == 0 and not (np.array_equal(o4["vec"][2], o1["vec"][0]) and
                                                      np.array_equal(o4["grid"][2], o1["grid"][0])):
                diff = t
            o4, rew, term, trunc, info = d4.step(acts[t])
            if term[2] or trunc[2]:
                ends.append((t, dict(zip(INFO_FIELDS, info[2].tolist())), float(d4.parts[2, d4.reward_names.index("win")]),
                             d4.view(o4["vec"][2], "self"),
                             d4.view(o4["vec"][2], "global")["time"][0]))
            if d1_ends == 0:
                o1, _, t1, tr1, _ = d1.step(acts[t][2:3])
                d1_ends += int(t1[0] or tr1[0])
        check("isolated: battle 2 of an n=4 driver == the same seed alone (until its game over)", diff is None,
              "" if diff is None else f"differs from step {diff}")
        ok = bool(ends)
        if ends:
            t, inf, r, me, time = ends[0]
            exp = (inf["score_own"] > inf["score_opp"]) - (inf["score_own"] < inf["score_opp"])
            fresh = me["alive"][0] == 1 and me["hp"][0] == 1 and me["gems"][0] == 0 and 0 < time < 0.05
            ok = inf["seed"] == 23 and inf["result"] == exp and r == exp and fresh
            detail = f"step {t}: {inf}, reward {r}, next obs: time {time:.3f}, alive/full hp/no gems {fresh}"
        else:
            detail = "battle 2 did not end within 800 steps"
        check("auto-reset: final info/reward of the finished battle, next obs = start of a new battle", ok, detail)
        nxt = [e[1] for e in ends[1:]]
        if nxt:
            check("auto-reset: the next battle in that row uses seed + n", nxt[0]["seed"] == 27, str(nxt[0]))


def test_memory():
    rng = np.random.default_rng(1)
    with VecDriver(n=16, seed=100) as d:
        episodes = 0
        while episodes < 100:
            _, _, term, trunc, _ = d.step(random_actions(rng, 16))
            episodes += int((term | trunc).sum())
        d.close()
        rss = [int(m) for line in d.log for m in re.findall(r"rss=(\d+)KB", line)]
    check("memory: RSS after 100 episodes within 3 MB of the start", len(rss) == 2 and rss[1] - rss[0] < 3000,
          f"start {rss[0] if rss else '?'} KB, after {episodes} episodes {rss[-1] if rss else '?'} KB")


def test_error():
    try:
        VecDriver(n=2, loc="NoSuchMap")
        check("errors: unknown map rejected", False, "driver started")
    except RuntimeError as e:
        check("errors: unknown map rejected with the driver's message", "NoSuchMap" in str(e), str(e).splitlines()[-1])


if __name__ == "__main__":
    for s in (1, 3, 5):
        test_same_as_serve(seed=s)
    test_same_as_serve(seed=2, loc="Ball1")
    from eval import gems
    test_same_as_serve(seed=4, policy=gems)        # picks up / carries / drops gems, fights
    test_deterministic()
    test_isolated_and_reset()
    test_memory()
    test_error()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
