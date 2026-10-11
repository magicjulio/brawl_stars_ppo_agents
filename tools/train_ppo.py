#!/usr/bin/env python3
"""Train a Gem Grab agent against the game's own bots with PPO (Proximal Policy Optimization), on the laptop's CPU.

Run it with the shared venv that already has torch:
    ../../.venv/bin/python tools/train_ppo.py                       # new run in runs/<date-time>/
    ../../.venv/bin/python tools/train_ppo.py --resume runs/X/last.pt
    ../../.venv/bin/python tools/train_ppo.py --eval runs/X/last.pt --eval-episodes 128     # just measure a model
    Ctrl-C (or `kill <pid>`) stops cleanly: the model is saved to last.pt first.

The policy steers our whole team (--agents 3, default): the same network plays each of our three brawlers, each
from its own point of view (one row per brawler: 256 battles x 3 = 768 rows); only the enemy team are bots.
--agents 1 steers player 0 only, with two bot teammates (the setting of the first runs).

How PPO works, in five lines (each step is one function below):
  1. collect()       play all 768 rows (brawlers) for T decisions each with the current policy; remember everything
  2. advantages()    for every decision, ask "did things go better or worse than the value net expected?"
  3. ppo_update()    nudge the policy towards decisions that went better, away from those that went worse,
                     but only a little ("clipping"), so one batch of luck cannot wreck the policy
  4. the value net learns at the same time to predict the discounted future reward (it is the "expected" in 2.)
  5. repeat with the improved policy. Data is used for a few epochs, then thrown away (PPO is "on-policy").

Log columns (one line per update; also in runs/<run>/log.csv):
  steps       decisions taken so far (one decision = 4 game ticks = 0.2 s of game time per battle)
  sps         decisions per second, whole loop (collect + update)
  coll/upd    seconds spent playing (tablet busy) and learning (laptop busy) in this update
  games       battles finished in this rollout; win%  share of them won (the policy plays with exploration noise)
              (one per battle, also with several brawlers per battle)
  gdiff       mean final gem difference (own - enemy); ret  mean episode return under the reward weights
              (without the wasted-shot cost)
  waste%      share of attack presses that were wasted shots (ammo loaded, no visible enemy in range)
  pi_loss     policy loss (noisy, its trend says little); v_loss  value prediction error
  ent         entropy = how random the policy still is (summed over the 4 action heads; starts at its maximum,
              4.9 with --aim auto, 8.1 with --aim free; tools/policy_stats.py shows it per head)
  kl          how far this update moved the policy (approx. KL divergence); ~0.01-0.03 is healthy
  clip        share of samples where the clip was active; ~0.1-0.2 is healthy
  ev          explained variance of the value net: 1 = predicts returns perfectly, 0 = no better than a constant
Every --eval-every updates the policy also plays the fixed eval battles (tools/eval.py, seeds 1_000_000+k);
baselines on those, steering all 3 (--agents 3): random 0%, scripted "gems" heuristic 27% wins; steering only
player 0 next to two bot teammates (--agents 1): random 17%, gems 72%, runs/auto15 74%.

Watching (needs no extra time from training: games are stored as seed + actions, a few KB each, and re-simulated
on the tablet only when you open one):
  every eval saves its battles to runs/<run>/eval/u<update>/s<seed>.npz and prints one won and one lost battle;
  --watch also saves every game of training battle 0 to runs/<run>/watch/ (with the value estimate on the timeline).
  Open one in the viewer:  python3 tools/episodes.py runs/<run>/eval/u0020/s1000003.npz
"""
import argparse
import csv
import dataclasses
import json
import pathlib
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bsvec import REWARD_PARTS  # noqa: E402
from bsvecenv import BrawlVecEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ============================================================================================================
# 1. Settings. Every number that shapes training lives here; all of them can be set on the command line.
# ============================================================================================================
@dataclasses.dataclass
class Config:
    total_steps: int = 200_000_000  # decisions in total (over all battles); ~4-5 h
    drivers: int = 8                # tablet processes (one per core)
    agents: int = 3                 # brawlers of our team the policy steers (1 = only player 0, two bot teammates)
    battles: int = 32               # battles per driver -> 8 * 32 = 256 battles, 768 brawlers (rows) in parallel
    repeat: int = 4                 # game ticks per decision (20 ticks = 1 s)
    rollout: int = 64               # T: decisions per brawler per update -> 768 * 64 = 49k samples per update
    gamma: float = 0.995            # discount: a reward 200 decisions (40 s) ahead counts ~37%
    gae_lambda: float = 0.95        # GAE: blend of short (biased) and long (noisy) advantage estimates
    lr: float = 3e-4                # Adam learning rate (decays linearly to 0 over total_steps)
    epochs: int = 3                 # passes over each rollout
    minibatch: int = 4096           # samples per gradient step
    clip: float = 0.2               # PPO clip: the new/old probability ratio is kept within 1 +- clip
    vf_coef: float = 0.5            # weight of the value loss next to the policy loss
    ent_coef: float = 0.01          # entropy bonus: keeps the policy exploring
    max_grad_norm: float = 0.5      # gradient clipping (stops single huge updates)
    target_kl: float = 0.05         # stop the epochs early if the policy moved further than this
    threads: int = 4                # torch CPU threads: the laptop has 4 real cores; 8 (hyperthreads) is ~3x slower
    eval_every: int = 20            # updates between eval runs (~2 min of training; 0 = never)
    eval_episodes: int = 32         # fixed battles per eval (~15 s). Few, so +-15% noise: watch the trend, not one value
    watch: bool = False             # record battle 0 for the viewer (runs/<run>/watch/, see tools/episodes.py)
    # what the agent can do and what it is rewarded for (the other reward pieces: bsvec.REWARD_WEIGHTS)
    aim: str = "auto"               # attack/super menu: "auto" = nothing or auto-aim; "free" = also 8 fixed directions
    dmg_reward: float = 0.05        # per 1000 damage dealt to enemy brawlers (a Shelly hit does ~600-1500)
    kill_reward: float = 0.1        # per enemy brawler killed
    waste_cost: float = 0.01        # pressing attack with ammo but no visible enemy in range = a wasted shot
    # who we play against (tools/eval.py's baselines above are on diff 0, random bot brawlers)
    diff: str = "0"                 # bot difficulty per tablet process, comma list cycled over the drivers: "2,3,4,4"
                                    # trains a quarter of the battles at 2, a quarter at 3, half at 4. The game's bots
                                    # get much stronger up to 4 (team45: 92% wins at 0, 38% at 2, 16% at 4), not after
    hero: str = ""                  # player 0's brawler (internal name); "" = ShotgunGirl (Shelly), the driver default
    mates: str = ""                 # our teammates' brawlers, comma list in player order (internal names: ShotgunGirl =
                                    # Shelly, Gunslinger = Colt, TntDude = Dynamike); "" = the bots' random picks
    bots: str = ""                  # brawlers the bots may pick (internal names); "" = any. Without --mates our
                                    # teammates are bot picks too: "--bots ShotgunGirl" alone = all six Shelly (hard20)
    eval_diff: int = -1             # bot difficulty of the evals; -1 = the highest one in --diff
    # one network per brawler instead of one shared network: "ShotgunGirl=runs/a/last.pt,Gunslinger=runs/b/last.pt"
    # (start checkpoints; each brawler's rows are played by and train only its own net, see TeamNet)
    team: str = ""
    freeze: str = ""                # brawlers whose net does not learn (comma list), e.g. a finished Shelly
    minutes: float = 0              # stop after this many minutes (0 = run until --total-steps)
    seed: int = 0

    def diffs(self):
        return [int(d) for d in self.diff.split(",")]

    def lineup_keys(self):
        """battle keys for who plays: --hero, --mates, --bots (only the ones that are set)"""
        return {k: v for k, v in (("hero", self.hero), ("mates", self.mates), ("bots", self.bots)) if v}

    def battle_keys(self):
        """battle keys for the evals: --eval-diff and the lineup"""
        return {"diff": self.eval_diff if self.eval_diff >= 0 else max(self.diffs()), **self.lineup_keys()}

    def slot_names(self):
        """brawler of each steered slot (player 0, then the teammates), for the per-brawler log columns"""
        mates = self.mates.split(",") if self.mates else ["?"]
        return [self.hero or "ShotgunGirl"] + [mates[k % len(mates)] for k in range(self.agents - 1)]


# ============================================================================================================
# 2. Actions. The env wants 10 floats per battle (move x/y, attack, aim x/y, super, aim x/y, gadget, unused).
#    The policy instead picks from four small menus ("heads"); each choice maps to fixed floats below.
#    Discrete menus are easy to learn with PPO and easy to read in the logs.
# ============================================================================================================
N_DIRS = 16
_angles = 2 * np.pi * np.arange(N_DIRS) / N_DIRS
_dirs = np.stack([np.cos(_angles), np.sin(_angles)], 1).astype(np.float32)     # [16, 2] unit vectors

# move: 16 directions + stand still
MOVE = np.concatenate([_dirs, np.zeros((1, 2), np.float32)])                   # [17, 2] -> (mx, my)
# attack and super -> (button, aim x, aim y). Button 1 is the game's own auto-aim: at the closest visible enemy in
# range, or straight ahead when nobody is in range. Button 2 aims at full range along (aim x, aim y).
#   "auto": nothing or auto-aim (2 entries). Easy to learn: only *when* to shoot, the game does the aiming.
#   "free": also aimed in one of 8 directions (10 entries): can shoot into bushes or ahead of a runner, but has to
#           learn aiming from scratch (the first run never did: its aimed shots were no better than chance).
_aim8 = _dirs[::2]                                                               # every second direction: 8
AIM_MENUS = {"auto": np.array([[0, 0, 0], [1, 0, 0]], np.float32),
             "free": np.concatenate([[[0, 0, 0], [1, 0, 0]],
                                     np.concatenate([np.full((8, 1), 2), _aim8], 1)]).astype(np.float32)}
GADGET = np.array([0, 1], np.float32)


def set_aim(mode):
    """Pick the attack/super menu (Config.aim). Call before building the network: it sets the head sizes."""
    global AIM, HEADS
    AIM = AIM_MENUS[mode]
    HEADS = {"move": len(MOVE), "attack": len(AIM), "super": len(AIM), "gadget": len(GADGET)}   # auto: 17, 2, 2, 2


set_aim("auto")


def to_env_actions(choice):
    """choice: int [rows, 4] (one column per head) -> float32 [rows, 10] in the env's format"""
    a = np.zeros((len(choice), 10), np.float32)
    a[:, 0:2] = MOVE[choice[:, 0]]
    a[:, 2:5] = AIM[choice[:, 1]]
    a[:, 5:8] = AIM[choice[:, 2]]
    a[:, 8] = GADGET[choice[:, 3]]
    return a


# ============================================================================================================
# 3. The network: observation -> (scores for every menu entry, value estimate).
#    Inputs:  vec  [B, 223] floats (own hero, allies, enemies, gems, projectiles, ... relative to the own hero)
#             grid [B, 21, 21] tiles around the own hero, each a bit set (wall, water, bush, ...)
#    The 6 "hero" fields in vec are brawler ids (categories, not amounts), so they go through an embedding.
#    One shared body feeds both the policy heads and the value head (cheaper than two networks).
# ============================================================================================================
class Net(nn.Module):
    def __init__(self, env):
        super().__init__()
        # which columns of vec hold brawler ids: self.hero, ally0..1.hero, enemy0..2.hero
        cols = []
        for name in ("self", "ally", "enemy"):
            b = env.blocks[name]
            nf = len(b.fields)
            cols += [b.start + k * nf + b.fields.index("hero") for k in range(b.count)]
        self.register_buffer("hero_cols", torch.tensor(cols))
        keep = torch.ones(env.obs_dim)
        keep[cols] = 0
        self.register_buffer("keep", keep)                       # zeroes the id columns in the float input
        self.register_buffer("bits", torch.arange(6, dtype=torch.uint8))   # grid flag bits 0..5

        self.hero_embed = nn.Embedding(1024, 8)                  # brawler id -> 8 learned numbers
        # The grid is always centred on the own hero, so "wall 2 tiles left of me" is always the same input: a plain
        # linear layer over all 6 x 21 x 21 bits learns such features directly. (A CNN made the update 2.5x slower on
        # this CPU; the update, not the tablet, would then limit training.)
        self.grid_net = nn.Sequential(nn.Flatten(), nn.Linear(6 * 21 * 21, 128), nn.ReLU())
        n_in = env.obs_dim + len(cols) * 8 + 128
        self.body = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.head_sizes = list(HEADS.values())
        self.policy_head = nn.Linear(256, sum(self.head_sizes))  # one score per menu entry, split into the 4 menus
        self.value_head = nn.Linear(256, 1)
        # small initial policy weights: every choice starts (almost) equally likely
        nn.init.orthogonal_(self.policy_head.weight, 0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, vec, grid):
        """vec float32 [B, 223], grid uint8 [B, 21, 21] -> (logits: list of 4 [B, k], value [B])"""
        heroes = vec[:, self.hero_cols].long().clamp(0, 1023)
        x = (vec * self.keep).clamp(-10, 10)
        g = ((grid.unsqueeze(1) >> self.bits.view(1, 6, 1, 1)) & 1).float()   # bit set -> 6 yes/no channels
        h = torch.cat([x, self.hero_embed(heroes).flatten(1), self.grid_net(g)], 1)
        h = self.body(h)
        logits = torch.split(self.policy_head(h), self.head_sizes, 1)
        return logits, self.value_head(h).squeeze(1)


class TeamNet(nn.Module):
    """One Net per brawler (--team): every row is played by the net of its own brawler, picked by the observation's
    own brawler id (self.hero), so the same model works in training, evals and play.py. Each net learns only from its
    own brawler's rows. Brawlers without a net of their own are played by the first one."""

    def __init__(self, env, names):
        super().__init__()
        self.names = list(names)                                 # brawler (internal name) of each net
        self.nets = nn.ModuleList(Net(env) for _ in self.names)
        self.head_sizes = self.nets[0].head_sizes
        self.hero_col = int(self.nets[0].hero_cols[0])           # vec column of the own brawler's id
        self.register_buffer("route", torch.zeros(1024, dtype=torch.long))   # brawler id -> index into nets

    def set_route(self, vec, slot_names):
        """learn the brawler ids from observations: vec rows 0..k-1 are battle 0's steered brawlers slot_names"""
        for k, name in enumerate(slot_names):
            if name in self.names:
                self.route[int(vec[k, self.hero_col])] = self.names.index(name)

    def forward(self, vec, grid):
        which = self.route[vec[:, self.hero_col].long().clamp(0, 1023)]
        rows = [r for r in (torch.nonzero(which == k).squeeze(1) for k in range(len(self.nets)))]
        outs = [net(vec[r], grid[r]) for net, r in zip(self.nets, rows) if len(r)]
        back = torch.argsort(torch.cat([r for r in rows if len(r)]))   # puts the rows back into the input order
        logits = [torch.cat([o[0][h] for o in outs])[back] for h in range(len(self.head_sizes))]
        return logits, torch.cat([o[1] for o in outs])[back]


def team_spec(spec):
    """'ShotgunGirl=runs/a/last.pt,Gunslinger=runs/b/last.pt' -> [(name, path), ...]"""
    return [tuple(p.split("=", 1)) for p in spec.split(",") if p]


def load_team(net, spec):
    """fill a TeamNet's nets from single-brawler checkpoints (or take that brawler's net out of a team checkpoint)"""
    for k, (name, path) in enumerate(team_spec(spec)):
        ck = torch.load(path)
        state = ck["net"]
        if "team" in ck:                                         # a team checkpoint: its net for this brawler
            j = ck["team"].index(name) if name in ck["team"] else 0
            state = {key[len(f"nets.{j}."):]: v for key, v in state.items() if key.startswith(f"nets.{j}.")}
        net.nets[k].load_state_dict(state)
        print(f"  {name}: net from {path} ({ck.get('steps', 0):,} steps)")


def build_net(ck, env):
    """the network of a checkpoint (a Net, or a TeamNet from a --team run), weights loaded. env: anything with
    .blocks and .obs_dim (the observation layout)"""
    set_aim(ck["config"].get("aim", "free"))                     # runs before the aim setting used the free menu
    net = TeamNet(env, ck["team"]) if "team" in ck else Net(env)
    net.load_state_dict(ck["net"])
    return net


def sample(logits):
    """Pick one entry per head at random, with the policy's probabilities. -> choice [B, 4], log-probability [B]"""
    choice, logp = [], 0
    for lg in logits:
        c = torch.multinomial(F.softmax(lg, 1), 1)               # [B, 1]
        logp = logp + F.log_softmax(lg, 1).gather(1, c).squeeze(1)
        choice.append(c)
    return torch.cat(choice, 1), logp


def logp_and_entropy(logits, choice):
    """log-probability of given choices [B, 4] under the policy, and the policy's entropy (both summed over heads)"""
    logp, ent = 0, 0
    for k, lg in enumerate(logits):
        lp = F.log_softmax(lg, 1)
        logp = logp + lp.gather(1, choice[:, k:k + 1]).squeeze(1)
        ent = ent + -(lp.exp() * lp).sum(1)
    return logp, ent


# ============================================================================================================
# 4. Rollout storage: T decisions x N battles of everything PPO needs later.
# ============================================================================================================
class Rollout:
    def __init__(self, T, N, obs_dim, grid):
        self.vec = np.zeros((T, N, obs_dim), np.float32)
        self.grid = np.zeros((T, N, grid, grid), np.uint8)
        self.choice = np.zeros((T, N, len(HEADS)), np.int64)
        self.logp = np.zeros((T, N), np.float32)        # log-probability of the choice when it was made
        self.value = np.zeros((T, N), np.float32)       # value net's estimate at that moment
        self.reward = np.zeros((T, N), np.float32)
        self.term = np.zeros((T, N), bool)              # battle ended after this decision (win/loss/draw)
        self.trunc = np.zeros((T, N), bool)             # battle cut off (only when a tablet process crashed)
        self.last_value = np.zeros(N, np.float32)       # value of the observation after the last decision


class Agent:
    """The network plus the bookkeeping to act on numpy observations."""

    def __init__(self, net):
        self.net = net

    @torch.inference_mode()
    def act(self, vec, grid):
        logits, value = self.net(torch.from_numpy(vec), torch.from_numpy(grid))
        choice, logp = sample(logits)
        return choice.numpy(), logp.numpy(), value.numpy()

    def eval_policy(self, obs, d):
        """policy in tools/eval.py's format: act(obs, d) -> env actions [d.n, 10]"""
        choice, _, _ = self.act(obs["vec"], obs["grid"])
        return to_env_actions(choice)


def wasted_shots(env, vec, choice):
    """-> (pressed, wasted): the living hero pressed attack; ...with at least one shot loaded but no visible enemy within
    attack range. (Hidden enemies in bushes don't count, so this also discourages shooting into bushes; small cost.)"""
    S, E = env.view(vec, "self"), env.view(vec, "enemy")
    loaded = S["ammo"][:, 0] * 3 >= 1                          # ammo = shots / 3
    dist = np.hypot(E["dx"], E["dy"])
    in_range = ((E["present"] > 0) & (E["alive"] > 0) & (E["visible"] > 0) & (dist <= S["range"])).any(1)
    pressed = (AIM[choice[:, 1], 0] > 0) & (S["alive"][:, 0] > 0)
    return pressed, pressed & loaded & ~in_range


DMG_PART, DEATH_PART = REWARD_PARTS.index("dmg_dealt"), REWARD_PARTS.index("death")   # columns of info["parts"]


def collect(env, agent, buf, obs, stats, value_now, waste_cost, diff_of):
    """Play until every battle has made T decisions. obs: the latest observation of every battle (updated in place).

    The env is asynchronous: each group of 48 battles answers on its own, and we act on whichever groups are ready,
    so the tablet never waits for the slowest group. A group that has its T decisions stops until the next rollout.
    """
    T, N = buf.reward.shape
    t_of = np.zeros(N, np.int64)                      # decisions recorded so far, per battle
    wasted = np.zeros(N, bool)                        # the pending decision of each battle wasted a shot

    def decide(ids, vec, grid):
        choice, logp, value = agent.act(vec, grid)
        t = t_of[ids]
        buf.vec[t, ids], buf.grid[t, ids], buf.choice[t, ids] = vec, grid, choice
        buf.logp[t, ids], buf.value[t, ids] = logp, value
        value_now[ids] = value
        pressed, wasted[ids] = wasted_shots(env, vec, choice)
        stats["presses"] += int(pressed.sum())
        stats["wasted"] += int(wasted[ids].sum())
        env.send(to_env_actions(choice), ids)

    decide(np.arange(N), obs["vec"], obs["grid"])
    in_flight = N
    while in_flight:
        o, rew, term, trunc, info = env.recv()
        ids = info["env_id"]
        t = t_of[ids]
        buf.reward[t, ids], buf.term[t, ids], buf.trunc[t, ids] = rew - waste_cost * wasted[ids], term, trunc
        t_of[ids] += 1
        obs["vec"][ids], obs["grid"][ids] = o["vec"], o["grid"]
        in_flight -= len(ids)

        # finished battles (crashes say nothing about the policy); a battle's brawlers share the result: count it once
        ended = info["final"] & ~info["crashed"] & (ids % env.agents == 0)
        stats["result"] += info["result"][ended].tolist()
        stats["diff"] += diff_of[ids][ended].tolist()
        stats["gdiff"] += (info["score_own"][ended] - info["score_opp"][ended]).tolist()
        stats["ret"] += info["episode_return"][ended].tolist()
        stats["crashed"] += int(info["crashed"].sum())
        slot = ids % env.agents                       # which of our brawlers each row is (player 0, teammate 1, 2)
        np.add.at(stats["slot_steps"], slot, 1)
        np.add.at(stats["slot_dmg"], slot, info["parts"][:, DMG_PART])
        np.add.at(stats["slot_deaths"], slot, info["parts"][:, DEATH_PART])

        more = t_of[ids] < T                          # whole groups always share the same count
        if more.all():
            decide(ids, o["vec"], o["grid"])
            in_flight += len(ids)
        else:                                         # this group is done: value of where it stands, for step 2
            _, _, buf.last_value[ids] = agent.act(o["vec"], o["grid"])


# ============================================================================================================
# 5. Advantages with GAE (Generalized Advantage Estimation).
#    delta_t = r_t + gamma * V(s_t+1) - V(s_t)        "one step of surprise": better (>0) or worse than expected
#    A_t     = delta_t + gamma * lambda * A_t+1       sum of future surprises, fading with gamma * lambda
#    return_t = A_t + V(s_t)                          the target the value net learns to predict
# ============================================================================================================
def advantages(buf, gamma, lam):
    T = len(buf.reward)
    adv = np.zeros_like(buf.reward)
    next_adv = np.zeros_like(buf.last_value)
    for t in reversed(range(T)):
        next_value = buf.value[t + 1] if t + 1 < T else buf.last_value
        # battle ended: nothing comes after a win/loss (value 0); after a crash cut-off we don't know what would have
        # come, so we assume "as expected" (V(s_t) itself). Either way the next row is already a new battle.
        next_value = np.where(buf.term[t], 0.0, np.where(buf.trunc[t], buf.value[t], next_value))
        delta = buf.reward[t] + gamma * next_value - buf.value[t]
        not_done = ~(buf.term[t] | buf.trunc[t])
        adv[t] = next_adv = delta + gamma * lam * not_done * next_adv
    return adv, adv + buf.value


# ============================================================================================================
# 6. The PPO update.
#    ratio = p_new(choice) / p_old(choice). The policy loss rewards raising the probability of choices with positive
#    advantage, but stops caring once the ratio leaves [1 - clip, 1 + clip]: small, safe steps.
# ============================================================================================================
def ppo_update(net, opt, buf, adv, ret, cfg):
    T, N = buf.reward.shape
    S = T * N
    vec = torch.from_numpy(buf.vec.reshape(S, -1))
    grid = torch.from_numpy(buf.grid.reshape(S, *buf.grid.shape[2:]))
    choice = torch.from_numpy(buf.choice.reshape(S, -1))
    old_logp = torch.from_numpy(buf.logp.reshape(S))
    adv = torch.from_numpy(adv.reshape(S))
    ret = torch.from_numpy(ret.reshape(S))
    log = {"pi_loss": [], "v_loss": [], "ent": [], "kl": [], "clip": []}
    for _ in range(cfg.epochs):
        perm = torch.randperm(S)
        for i in range(0, S, cfg.minibatch):
            mb = perm[i:i + cfg.minibatch]
            logits, value = net(vec[mb], grid[mb])
            logp, ent = logp_and_entropy(logits, choice[mb])
            a = adv[mb]
            a = (a - a.mean()) / (a.std() + 1e-8)                   # normalised: the scale of rewards stops mattering
            ratio = (logp - old_logp[mb]).exp()
            pi_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            v_loss = 0.5 * (value - ret[mb]).pow(2).mean()
            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                log["pi_loss"].append(pi_loss.item())
                log["v_loss"].append(v_loss.item())
                log["ent"].append(ent.mean().item())
                log["kl"].append(((ratio - 1) - ratio.log()).mean().item())   # approx. KL(old || new)
                log["clip"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        if np.mean(log["kl"][-(S // cfg.minibatch):]) > cfg.target_kl:
            break                                                    # moved far enough for this batch
    out = {k: float(np.mean(v)) for k, v in log.items()}
    out["ev"] = float(1 - np.var(buf.value.ravel() - ret.numpy()) / (np.var(ret.numpy()) + 1e-8))
    return out


# ============================================================================================================
# 7. Main loop.
# ============================================================================================================
def run_eval(agent, cfg, out_dir):
    """Play the fixed eval battles, save each to out_dir (for the viewer). -> (stats, one won + one lost battle file)"""
    from eval import evaluate                       # imported late: only needed when evaluating
    s = evaluate(agent.eval_policy, episodes=cfg.eval_episodes, n=min(32, cfg.eval_episodes), quiet=True,
                 agents=cfg.agents, **cfg.battle_keys())
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = {}
    for ep in s["episode_list"]:
        path = out_dir / f"s{ep.seed}.npz"
        ep.save(path)
        examples.setdefault("won" if ep.result > 0 else "lost" if ep.result < 0 else "draw", path)
    return s, examples


def eval_checkpoint(path, cfg, save_team=None):
    """Play the fixed eval battles with a saved model; save them next to it (eval_<name>/) for the dashboard.
    --agents decides the setting: 3 = it steers the whole team, 1 = it plays with two bot teammates.
    path may also be a team spec "ShotgunGirl=a.pt,Gunslinger=b.pt,..." (one net per brawler; --save-team keeps it)."""
    from eval import HEADER, evaluate, fmt
    torch.set_num_threads(cfg.threads)
    agent = None

    def policy(obs, d):                             # the network is built on the first call: it needs d's layout
        nonlocal agent
        if agent is None:
            if "=" in str(path):                      # a team put together from single-brawler checkpoints
                spec = team_spec(path)
                set_aim(torch.load(spec[0][1])["config"].get("aim", "free"))
                net = TeamNet(d, [name for name, _ in spec])
                load_team(net, path)
                net.set_route(obs["vec"], cfg.slot_names())
                if save_team:                         # keep it as one checkpoint (for play.py --vs)
                    torch.save({"net": net.state_dict(), "team": net.names, "steps": 0, "update": 0,
                                "config": {**dataclasses.asdict(cfg), "team": path}}, save_team)
            else:
                net = build_net(torch.load(path), d)
            agent = Agent(net)
        return agent.eval_policy(obs, d)

    keys = cfg.battle_keys()
    s = evaluate(policy, episodes=cfg.eval_episodes, n=min(64, cfg.eval_episodes), agents=cfg.agents, **keys)
    path = pathlib.Path(save_team or path)
    out = path.parent / (f"eval_{path.stem}" + ("" if cfg.agents == 3 else f"_agents{cfg.agents}")
                         + (f"_diff{keys['diff']}" if keys["diff"] else "") + (f"_hero{cfg.hero}" if cfg.hero else "")
                         + (f"_mates{cfg.mates}" if cfg.mates else "")
                         + (f"_bots{cfg.bots}" if cfg.bots else ""))
    for ep in s["episode_list"]:
        ep.save(out / f"s{ep.seed}.npz")
    print(f"{cfg.agents} steered, player 0 {cfg.hero or 'ShotgunGirl'}, teammates {cfg.mates or 'bot picks'}, "
          f"bots: diff {keys['diff']}, "
          f"brawlers {cfg.bots or 'any'}")
    print(HEADER)
    print(fmt(path.stem, s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--resume", help="checkpoint (.pt) to continue from")
    ap.add_argument("--run", help="run directory (default runs/<date-time>)")
    ap.add_argument("--eval", metavar="CHECKPOINT", help="only evaluate a saved model (--eval-episodes battles); "
                    "or a team spec Name=a.pt,Name=b.pt,... (one net per brawler)")
    ap.add_argument("--save-team", metavar="PATH", help="with --eval <team spec>: save the team as one checkpoint")
    args = ap.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    if args.eval:
        return eval_checkpoint(args.eval, cfg, args.save_team)

    # Ctrl-C and `kill <pid>` both stop training cleanly (model saved), also when started in the background
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    run_dir = pathlib.Path(args.run) if args.run else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))   # for the dashboard

    ck = torch.load(args.resume) if args.resume else None
    if ck or cfg.team:                                # the network's menu sizes come with the checkpoint
        cfg.aim = (ck or torch.load(team_spec(cfg.team)[0][1]))["config"].get("aim", "free")
    set_aim(cfg.aim)
    diffs = [cfg.diffs()[i % len(cfg.diffs())] for i in range(cfg.drivers)]     # one bot difficulty per driver
    env = BrawlVecEnv(drivers=cfg.drivers, n=cfg.battles * cfg.agents, repeat=cfg.repeat, agents=cfg.agents,
                      battle=[{"diff": d} for d in diffs], **cfg.lineup_keys())
    diff_of = np.repeat(diffs, env.n)                 # bot difficulty of every row
    env.set_weights({"dmg_dealt": cfg.dmg_reward, "kill": cfg.kill_reward})
    N = env.num_envs
    names = ck.get("team") if ck else [name for name, _ in team_spec(cfg.team)] if cfg.team else None
    net = TeamNet(env, names) if names else Net(env)
    if names:
        for name, sub in zip(names, net.nets):
            sub.requires_grad_(name not in cfg.freeze.split(","))
        if not ck:
            load_team(net, cfg.team)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr, eps=1e-5)
    steps, update, best = 0, 0, -1.0
    if ck:
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        steps, update, best = ck["steps"], ck["update"], ck.get("best", -1.0)
        print(f"resumed {args.resume} at {steps:,} steps")
    agent = Agent(net)
    print(f"run {run_dir}  {N // cfg.agents} battles x {cfg.agents} brawlers  "
          f"{sum(p.numel() for p in net.parameters()):,} parameters")
    print(f"player 0 {cfg.hero or 'ShotgunGirl'}, teammates {cfg.mates or 'bot picks'}; bots: difficulty per driver {diffs}, brawlers {cfg.bots or 'any'}; "
          f"evals at {cfg.battle_keys()}")

    value_now = np.zeros(N, np.float32)               # latest value estimate per battle (for --watch)
    if cfg.watch:
        from episodes import EnvRecorder
        EnvRecorder(env, [0], out_dir=run_dir / "watch", extras={"value": lambda ids: value_now[ids]})

    def save(name):
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "update": update,
                    "best": best, "config": dataclasses.asdict(cfg), **({"team": names} if names else {})},
                   run_dir / name)

    buf = Rollout(cfg.rollout, N, env.obs_dim, env.grid_size)
    obs, _ = env.reset()
    if names:
        net.set_route(obs["vec"], cfg.slot_names())   # brawler id of each steered slot -> its net
    log_file = open(run_dir / "log.csv", "a", newline="")
    log_csv = None
    crashes_saved = 0
    print(f"{'steps':>12} {'sps':>6} {'coll':>5} {'upd':>5} {'games':>5} {'win%':>5} {'gdiff':>6} {'ret':>6} "
          f"{'waste%':>6} {'pi_loss':>8} {'v_loss':>7} {'ent':>5} {'kl':>6} {'clip':>5} {'ev':>5}")
    start_steps, start_time = steps, time.time()
    slots = cfg.slot_names()                          # brawler of each steered slot, for the per-brawler log line
    per_min = lambda decisions: max(decisions, 1) * cfg.repeat / 20 / 60   # decisions -> minutes (20 ticks/s)
    try:
        while steps < cfg.total_steps and not (cfg.minutes and time.time() - start_time > 60 * cfg.minutes):
            # learning rate falls linearly to 0 over this run (from where a --resume started, not from step 0: a
            # resumed run would otherwise begin with an already small rate)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * max(0.0, 1 - (steps - start_steps) / max(cfg.total_steps - start_steps, 1))
            t0 = time.perf_counter()
            stats = {"result": [], "diff": [], "gdiff": [], "ret": [], "crashed": 0, "presses": 0, "wasted": 0,
                     **{k: np.zeros(cfg.agents) for k in ("slot_steps", "slot_dmg", "slot_deaths")}}
            collect(env, agent, buf, obs, stats, value_now, cfg.waste_cost, diff_of)
            t1 = time.perf_counter()
            adv, ret = advantages(buf, cfg.gamma, cfg.gae_lambda)
            upd = ppo_update(net, opt, buf, adv, ret, cfg)
            t2 = time.perf_counter()
            steps += cfg.rollout * N
            update += 1

            res = np.array(stats["result"])
            row = {"steps": steps, "sps": cfg.rollout * N / (t2 - t0), "coll": t1 - t0, "upd": t2 - t1,
                   "games": len(res), "win": 100 * (res > 0).mean() if len(res) else float("nan"),
                   "gdiff": np.mean(stats["gdiff"]) if len(res) else float("nan"),
                   "ret": np.mean(stats["ret"]) if len(res) else float("nan"), "crashed": stats["crashed"],
                   "waste": 100 * stats["wasted"] / max(stats["presses"], 1), **upd,
                   **{f"win_d{d}": 100 * (res[np.array(stats["diff"]) == d] > 0).mean()
                      if (np.array(stats["diff"]) == d).any() else float("nan") for d in sorted(set(diffs))},
                   # per steered brawler (slot 0 = player 0): damage dealt and deaths per minute of battle
                   **{f"dmg{k}": stats["slot_dmg"][k] * 1000 / per_min(stats["slot_steps"][k]) for k in range(cfg.agents)},
                   **{f"deaths{k}": stats["slot_deaths"][k] / per_min(stats["slot_steps"][k]) for k in range(cfg.agents)},
                   "eval_win": "", "eval_gdiff": "", "time": time.time()}
            print(f"{steps:>12,} {row['sps']:>6.0f} {row['coll']:>5.1f} {row['upd']:>5.1f} {row['games']:>5} "
                  f"{row['win']:>5.1f} {row['gdiff']:>+6.2f} {row['ret']:>+6.2f} {row['waste']:>6.1f} "
                  f"{upd['pi_loss']:>+8.4f} "
                  f"{upd['v_loss']:>7.4f} {upd['ent']:>5.2f} {upd['kl']:>6.4f} {upd['clip']:>5.3f} {upd['ev']:>5.2f}"
                  + (("  win by diff " + " ".join(f"{d}:{row[f'win_d{d}']:.0f}" for d in sorted(set(diffs))))
                     if len(set(diffs)) > 1 else "")
                  + (f"  crashed {stats['crashed']}" if stats["crashed"] else ""), flush=True)
            if len(set(slots)) > 1:                   # a mixed team: how each brawler does
                print("    dmg/min, deaths/min: " + "  ".join(
                    f"{name} {row[f'dmg{k}']:.0f} {row[f'deaths{k}']:.2f}" for k, name in enumerate(slots)), flush=True)
            # a tablet process that crashed or hung was restarted by the env; keep its record for debugging
            for c in env.crashes[crashes_saved:]:
                with open(run_dir / "crashes.jsonl", "a") as f:
                    f.write(json.dumps({**c, "update": update}, default=str) + "\n")
                print(f"  driver {c['driver']} restarted: {c['reason']} (details in {run_dir / 'crashes.jsonl'})")
            crashes_saved = len(env.crashes)

            if cfg.eval_every and update % cfg.eval_every == 0:
                t3 = time.perf_counter()
                try:
                    s, examples = run_eval(agent, cfg, run_dir / "eval" / f"u{update:04d}")
                except RuntimeError as e:             # the eval driver crashed (rare libg faults): skip this eval,
                    print(f"  eval skipped, its driver crashed: {str(e)[:200]}", flush=True)   # keep training
                    with open(run_dir / "crashes.jsonl", "a") as f:
                        f.write(json.dumps({"eval": True, "update": update, "reason": str(e)}) + "\n")
                    s = None
            if cfg.eval_every and update % cfg.eval_every == 0 and s is not None:
                win, gdiff = s["win_rate"], s["gem_diff"]
                row["eval_win"], row["eval_gdiff"] = win, gdiff
                print(f"  eval: {100 * win:.1f}% wins, gem diff {gdiff:+.2f} over {cfg.eval_episodes} battles "
                      f"({time.perf_counter() - t3:.0f} s)" + ("  new best" if win > best else ""), flush=True)
                for kind in ("won", "lost"):
                    if kind in examples:
                        print(f"    watch a {kind} one: python3 tools/episodes.py {examples[kind]}")
                if cfg.watch:
                    watched = sorted((run_dir / "watch").glob("*.npz"), key=lambda f: f.stat().st_mtime)
                    if watched:
                        print(f"    latest training game: python3 tools/episodes.py {watched[-1]}")
                if win > best:
                    best = win
                    save("best.pt")
                save("last.pt")

            if log_csv is None:
                log_csv = csv.DictWriter(log_file, fieldnames=list(row))
                if log_file.tell() == 0:
                    log_csv.writeheader()
            log_csv.writerow(row)
            log_file.flush()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save("last.pt")
        print(f"saved {run_dir / 'last.pt'}")
        log_file.close()
        env.close()


if __name__ == "__main__":
    main()
