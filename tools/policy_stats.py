"""What does a trained policy actually do? Plays battles with a checkpoint on one driver and measures behaviour.

    ../../.venv/bin/python tools/policy_stats.py runs/X/last.pt [--steps 600] [--n 96] [--diff 4]
        [--hero ShotgunGirl] [--mates Gunslinger,TntDude]

Prints how decided each action head is (entropy vs uniform), when it presses attack (enemy in range or not),
how well aimed shots point at the nearest visible enemy, when it uses super and gadget, and how much time it spends
near the gem mine. With a mixed team (--hero/--mates) the super and gadget lines are split per brawler.
"""
import argparse, sys, pathlib
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bsvec import VecDriver
from eval import _nearest_enemy, gems
import train_ppo as tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n", type=int, default=96, help="rows (brawlers)")
    ap.add_argument("--agents", type=int, help="brawlers steered per battle (default: the run's setting)")
    ap.add_argument("--diff", type=int, default=0, help="bot difficulty")
    ap.add_argument("--hero", default="", help="player 0's brawler (internal name)")
    ap.add_argument("--mates", default="", help="our teammates' brawlers (internal names, comma list)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ent, choices, ctx = [], [], []
    ck = torch.load(a.ckpt)
    agents = a.agents or ck["config"].get("agents", 1)
    keys = {k: v for k, v in (("hero", a.hero), ("mates", a.mates)) if v}
    with VecDriver(n=a.n, repeat=4, seed=2_000_000, loc="Gemgrab1", agents=agents, diff=a.diff, **keys) as d:
        net = tp.build_net(ck, d); net.eval()
        obs = d.obs
        for t in range(a.steps):
            vec, grid = obs["vec"].copy(), obs["grid"].copy()
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(vec), torch.from_numpy(grid))
            ent.append(np.stack([-(F.softmax(l, 1) * F.log_softmax(l, 1)).sum(1).numpy() for l in logits], 1))
            c = np.stack([torch.multinomial(F.softmax(l, 1), 1)[:, 0].numpy() for l in logits], 1)
            S, G = d.view(vec, "self"), d.view(vec, "global")
            dx, dy, dist, vis = _nearest_enemy(d, vec)
            ref = gems(obs, d)
            ctx.append(dict(alive=S["alive"][:, 0] > 0, empty=S["ammo"][:, 0] * 3 < 1, rng=S["range"][:, 0],
                            dx=dx, dy=dy, dist=dist, vis=vis, mine=np.hypot(G["mine_dx"][:, 0], G["mine_dy"][:, 0]),
                            ref_attack=ref[:, 2] > 0, ready=S["super_ready"][:, 0] > 0, srng=S["super_range"][:, 0],
                            slot=np.arange(len(vec)) % agents))
            choices.append(c)
            obs, *_ = d.step(tp.to_env_actions(c))
    ent, ch = np.concatenate(ent), np.concatenate(choices)
    X = {k: np.concatenate([c[k] for c in ctx]) for k in ctx[0]}
    alive = X["alive"]
    print(f"{alive.sum()} decisions while alive")
    print("\nhow decided is each head (entropy / uniform; 1.00 = still random):")
    for k, (name, size) in enumerate(tp.HEADS.items()):
        print(f"  {name:7s} {ent[alive, k].mean() / np.log(size):.2f}   choice shares: "
              + " ".join(f"{s:.2f}" for s in np.bincount(ch[alive, k], minlength=size) / alive.sum()))
    AIM = tp.AIM
    btn = AIM[ch[:, 1], 0]
    inr = X["vis"] & (X["dist"] <= X["rng"])
    print("\nattack button pressed (any):")
    for name, m in [("enemy visible & in range", alive & inr), ("no enemy in range", alive & ~inr)]:
        print(f"  {name:26s} {(btn[m] > 0).mean():.2f}  ({m.mean():.0%} of the time)")
    print(f"  gems heuristic would attack {X['ref_attack'][alive].mean():.0%} of the time")
    aimed = alive & (btn == 2) & X["vis"] & np.isfinite(X["dist"]) & (X["dist"] <= 1.5 * X["rng"])
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(AIM[ch[:, 1], 2], AIM[ch[:, 1], 1])
                                                 - np.arctan2(X["dy"], X["dx"]))))))
    if aimed.any():                               # only the "free" menu has aimed shots
        print(f"\naimed shots with a visible enemy within 1.5x range: n={aimed.sum()}, angle error to that enemy: "
              f"median {np.median(ang[aimed]):.0f} deg, within 23 deg: {(ang[aimed] <= 23).mean():.0%} "
              f"(random aim: median 90, within 23: ~12%)")
    share = (btn[alive & inr] == 1).mean() / max((btn[alive & inr] > 0).mean(), 1e-9)
    print(f"auto-aim share of presses with an enemy in range: {share:.0%}")
    # super and gadget: pressed when the super is charged, with or without a visible enemy within super range
    cfg = tp.Config(agents=agents, hero=a.hero, mates=a.mates)
    sup, gad = AIM[ch[:, 2], 0] > 0, ch[:, 3] > 0
    in_srng = X["vis"] & (X["dist"] <= X["srng"])
    groups = [("all", alive)] if not (a.hero or a.mates) else \
        [(name, alive & (X["slot"] == k)) for k, name in enumerate(cfg.slot_names())]
    print("\nsuper pressed while charged (enemy within super range / not), gadget pressed:")
    for name, m in groups:
        r = m & X["ready"]
        print(f"  {name:12s} super charged {r.mean() / max(m.mean(), 1e-9):.0%} of the time; pressed "
              f"{sup[r & in_srng].mean():.2f} with an enemy in range ({(r & in_srng).sum()} decisions), "
              f"{sup[r & ~in_srng].mean():.2f} without; gadget pressed {gad[m].mean():.2f}")
    # empty clip: is it dangerous? deaths within the next 3 s (15 decisions), empty+enemy in range vs all moments
    A = np.stack([c["alive"] for c in ctx])                     # [steps, n]
    E = np.stack([c["empty"] for c in ctx]) & A
    R = np.stack([c["vis"] & (c["dist"] <= c["rng"]) for c in ctx]) & A
    H = 15
    dies = np.zeros_like(A[:-H])
    for k in range(1, H + 1):
        dies |= ~A[k:len(A) - H + k]
    A0, E0, R0 = A[:-H], E[:-H], R[:-H]
    print(f"\nammo: empty {E[A].mean():.0%} of the time alive; empty with an enemy in range {(E & R)[A].mean():.0%}")
    print(f"  dies within 3 s: any moment {dies[A0].mean():.0%}, enemy in range {dies[R0].mean():.0%}, "
          f"enemy in range + empty {dies[E0 & R0].mean():.0%}")
    print(f"\ntime within 0.15 (1.5 tiles) of the gem mine: {(X['mine'][alive] < 0.15).mean():.0%}, "
          f"within 0.3: {(X['mine'][alive] < 0.3).mean():.0%}")


if __name__ == "__main__":
    main()
