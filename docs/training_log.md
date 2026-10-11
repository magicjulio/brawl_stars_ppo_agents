# Training log

Every training run, what changed, and what it taught us. Newest last. All runs: Gem Grab, map Gemgrab1, PPO from
`tools/train_ppo.py`, 1 decision = 4 game ticks = 0.2 s. "Eval" = the fixed eval battles of `tools/eval.py` (seeds
1_000_000+k, never used for training), 128 battles for final numbers (32 during training, about +-15% noise).
Measure any model: `../../.venv/bin/python tools/train_ppo.py --eval runs/X/last.pt --eval-episodes 128 [--diff 4 --mates ShotgunGirl]`.

## Baselines (scripted policies, `tools/eval.py --policy idle,random,chase,gems`)
| setting | idle | random | chase | gems |
|---|---|---|---|---|
| player 0 steered, 2 bot teammates (random brawlers), bots diff 0 | 15.6% | 17.2% | 71.9% | 71.9% |
| all 3 steered, bots diff 0 | | 0% | | 26.6% |
| player 0 steered, 2 Shelly bot teammates, bots diff 0 | 6.2% | | | |

## Lessons so far
1. **Make the useful action easy to find, and reward it directly.** The first run's free aiming (8 directions) never
   beat chance; the game's own auto-aim plus a 10x damage reward and a small cost for wasted shots took the win rate
   from 38% to 74% (first20 -> auto15).
2. **Steering the whole team** works with one shared network (each brawler sees the battle from its own position):
   59% after 10M decisions, 94.5% after 30M (team15 -> team45).
3. **Train against the opponent you care about.** All runs up to team45 played bot difficulty 0 (the driver's
   default). The game's bots get much stronger up to difficulty 4: team45 wins 92% at 0, 84% at 1, 38% at 2, 20% at
   3, 16% at 4, 11% at 6, 16% at 9 (64 battles each). Difficulty >= 3 also simulates ~1.8x slower. Difficulty 4 is the game's maximum: from 4 on the
   bots think every 6 ticks (20 below 3, 10 at 3) and always use their skills; nothing else depends on it (driver_plan.md).
4. **A narrow setting doesn't transfer.** hard20 learned to beat difficulty 4 when all six players were Shelly (23% ->
   69%) but did not improve at all against random enemy brawlers (24% -> 23%).
5. Resuming a run: the first update jumps (kl 0.3-0.5, value explained variance negative) and settles within a few
   updates. The learning rate decays over the resumed run since 2026-10-10 (before, it continued the old schedule
   and started small).

## Runs
| run | from | setting | decisions | time | final eval (128) |
|---|---|---|---|---|---|
| first20 | scratch | player 0 + 2 bot mates, diff 0, free aim | 10M | 21 min | 38.3% |
| auto15 | scratch | same, auto-aim menu, dmg reward 0.05, kill 0.1, waste cost 0.01 | 8M | 15 min | 74.2%, gem diff +6.3 |
| team15 | scratch | all 3 steered (mates random brawlers), diff 0 | 10M | 15 min | 58.6% (56.2% as player 0 with bot mates) |
| team45 | team15 | same | 30M total | +30 min | **94.5%** at diff 0 (79.7% as player 0 with bot mates); 23% at diff 4 |
| hard20 | team45 | all six Shelly (`--bots ShotgunGirl`), diff 2,2,3,3,4,4,4,4 | 40M total | 21 min | 68.8% at diff 4 all-Shelly; 22.7% diff 4 random enemies; 93.8% diff 0 |
| shelly30 | hard20 | our team Shelly (`--mates ShotgunGirl`), enemies random, diff 2,3,3,4,4,4,4,4 | 58M total | 32 min | **43.8%** at diff 4 (hard20 25.0%); 95.3% at diff 0 |
| shelly60 | shelly30 | same | 72M total | 25 min (stopped: flat) | 37.5% at diff 4 (= shelly30 within noise); 33.6% with lineup Shelly+Colt+Dynamike |
| team3_30 | shelly60 | our team Shelly+Colt+Dynamike (`--hero ShotgunGirl --mates Gunslinger,TntDude`), diff 3,4,4,4,4,4,4,4 | 85M total | ~19 min (ended by an eval driver crash) | 28.1% team lineup (start 33.6%); 20.3% as 3 Shelly (start 37.5%) |
| trio40 | shelly30 x3 | one net per brawler (`--team`), Shelly+Colt+Dynamike, diff 3,4,4,4,4,4,4,4 | 22.7M (new count) | 40 min | 31.2% [24-40] team lineup |
| trio_bull | trio40 (Shelly, Colt) + shelly30 (Bull) | `--team`, Shelly+Colt+Bull, diff 3,4,4,4,4,4,4,4 | 15.0M (new count) | 28 min | **36.7%** [29-45] team lineup |
| trio_bull2 | trio_bull | same | 22.6M (from 15.0M) | 22 min | **37.5%** [30-46] team lineup |

Details per run:
- **first20** (2026-10-09): training win 23% -> 38%, plateau after ~10 min. `tools/policy_stats.py` showed every
  action head still ~random, attack pressed 88% of the time regardless of enemies, aimed shots at chance. One driver
  crash (auto-restarted; reasons are now saved to runs/<run>/crashes.jsonl).
- **auto15** (2026-10-09): training win 43% -> 70%, wasted shots 31% -> 8%. Attacks 49% of steps with an enemy in range
  vs 23% without. Empty clip 51% of the time but only +3 points death risk, so no zero-ammo cost was added.
  Super/gadget heads still ~random.
- **team15** (2026-10-09): driver v5 `agents=3`. Training win 2% (first 2M) -> 34% (3M) -> 54% (5M) -> 63% (9M), still
  rising.
- **team45** (2026-10-09): evals 66 -> 75 -> 84 -> 91 -> 97%, then 88-97%. Final 121/7, gems 12.9 vs 0.9. Entropy
  still 4.0 of 4.9 at the end (the move head stays fairly random).
- **hard20** (2026-10-10): diff-4 evals (32 battles) 38 -> 41 -> 28 -> 50 -> 62 -> 44 -> 62 -> 72 -> 62 -> 69%; training
  wins at the end ~100% at diff 2, 60-75% at 3, 78-93% at 4. ~14k decisions/s.
- **shelly30** (2026-10-10): the setting from now on. Baselines there (diff 4, 64 battles): team45 25.0%, hard20 31.2%
  (128 battles: hard20 25.0%). Diff-4 evals (32 battles) 22 -> 28 -> 16 -> 31 -> 25 -> 28 -> ... -> 44 -> 34 -> 31 -> 47
  -> 44%: slow but steady, still rising at the end. Training wins at the end ~100% at diff 2, 30-65% at 3, 40-70% at 4.
  Entropy 3.84 -> 3.6. Only ~6.5-12k decisions/s: the laptop's update took 2.3-4 s instead of 1.8 s because a Firefox
  tab used ~170% CPU and the training process ran at nice 10 (started from a background session). Close heavy viewer
  tabs during runs.
- **shelly60** (2026-10-10): stopped after 25 of 60 minutes because nothing moved: training win at diff 4 41-48%,
  32-battle evals 25-50% the whole time. 128-battle eval 37.5% [30-46] vs shelly30's 43.8%: the Shelly-only team has
  plateaued at ~40% vs difficulty 4 with this recipe. Policy (tools/policy_stats.py --diff 4 --mates ShotgunGirl):
  super fired within ~2 decisions of charging, mostly with an enemy in range (0.85-0.93 vs 0.51-0.68 without);
  gadget pressed 47% of the time (fires at the first chance); attack 43% with an enemy in range, 7% without.
  Baseline for the mixed team: shelly60 with `--hero ShotgunGirl --mates Gunslinger,TntDude` at diff 4: 33.6% [26-42]
  (Colt and Dynamike never steered before).
- **team3_30** (2026-10-10): first mixed-team run, from shelly60. Training win at diff 4 flat at 35-41% for 13M
  decisions; 32-battle evals 22-44%. Per brawler (dmg/min, deaths/min, first -> last sixth of the run): Shelly 9.8k
  1.84 -> 8.0k 1.44, Colt 9.8k 1.43 -> 10.0k 1.48, Dynamike 6.0k 1.54 -> 3.7k 0.90: Dynamike learned to stay alive
  rather than to hit (a guess: his throws are slow, so auto-aimed throws miss moving targets). 128-battle evals: team lineup
  28.1% [21-37] (no gain over the 33.6% start), 3 Shelly 20.3% [14-28] (Shelly got worse: a third of the rows now,
  other teammates). Ended after ~19 min when an in-training eval driver segfaulted in
  LogicProjectileServer::handleCollisions (libg+0x418c84: a projectile's owner pointer +0x6c was null; never seen
  before; eval drivers had no restart). Fixed in train_ppo.py: a crashed eval is skipped and logged to crashes.jsonl.

- **Playtest** (2026-10-10, phone over Wi-Fi): the user (Shelly) with two difficulty-2 Shelly bots vs three shelly30
  agents (`play.py --vs runs/shelly30/last.pt --mates bots ... diff=2`). Of the finished battles the agents won ~8,
  the user 2 (e.g. 0-14, 1-12, 3-10, 0-12 vs 13-6, 15-1), several under heavy Wi-Fi lag. The user's verdict: "assume 3
  Shelly agents beat me consistently".
- **trio40** (2026-10-11): first run with one network per brawler (`--team`, train_ppo.TeamNet: each row is played by
  and trains only its own brawler's net, routed by the observation's own brawler id). All three nets started from
  shelly30. Training win at diff 4 by quarter 28 -> 32 -> 31 -> 35%; 32-battle evals 12-34%. Per brawler (dmg/min,
  deaths/min, first -> last update): Shelly 8.0k 2.4 -> 6.7k 1.2, Colt 10.1k 1.9 -> 9.2k 1.6, Dynamike 9.0k 3.0 ->
  4.5k 1.3: Dynamike again learned to hide instead of hitting (as in team3_30: auto-aimed slow throws at the target's
  current position miss moving enemies). 128-battle eval 31.2% (shelly30 with this lineup: 33.6%). Decision: replace
  Dynamike by Bull (`BullDude`, shotgun tank: auto-aim works, Shelly's play transfers).

- **trio_bull** (2026-10-11): Shelly + Colt + Bull, one net each (Shelly's and Colt's from trio40, Bull's from
  shelly30). Training win at diff 4 by quarter 35 -> 35 -> 34 -> 42%; 32-battle evals 19-38%. 128-battle eval 36.7%
  (gem diff +0.2), the best mixed team so far (trio40 31.2%, shelly30 with S+C+D 33.6%). Per brawler at the end:
  Shelly 8.4k dmg/min 1.5 deaths/min, Colt 10.7k 1.7, Bull only 1.7k 0.5: Bull stays back (fewest deaths, little
  damage) while Shelly and Colt fight; the team still wins more. Not studied further (out of time).
- **trio_bull2** (2026-10-11): trio_bull continued for 22 min (training win rising at the end of trio_bull).
  Training win at diff 4 by quarter 39 -> 33 -> 34 -> 36%; 32-battle evals 19-34%. 128-battle eval 37.5% (gem
  diff -0.35): the same as trio_bull within noise, so this lineup has plateaued at ~37% vs difficulty-4 bots with this
  recipe. Bull still deals little damage (2.2k/min, 0.5 deaths/min). (A first attempt died after 2 min without saving:
  the process was killed with its shell; long runs are now started with `setsid nohup ... & disown`.)

## Reward shaping: what the reward is made of (measured 2026-10-10)
Weights in training: win 1, gem_diff 0.1/gem, countdown 0.5, gems_carried 0.05/gem, kill 0.1 (`--kill-reward`),
death -0.1, dmg_dealt 0.05 per 1000 hp (`--dmg-reward`), dmg_taken -0.1 per life, low_hp -0.02/s, wasted shot -0.01.
Mean sum per battle for player 0 over 512 diff-4 eval battles of shelly30/shelly60 (weighted contribution in brackets;
the battles last ~580 decisions):

| | win | gem_diff | countdown | gems_carried | kill | death | dmg_dealt | dmg_taken | low_hp |
|---|---|---|---|---|---|---|---|---|---|
| won (195) | +1.00 | +14.9 (+1.49) | +1 (+0.50) | +5.6 (+0.28) | +2.9 (+0.29) | 2.3 (-0.23) | 17.2 (+0.86) | 4.5 (-0.45) | 2.8 (-0.06) |
| lost (317) | -1.00 | -9.2 (-0.92) | -1 (-0.50) | +1.1 (+0.06) | +0.9 (+0.09) | 2.9 (-0.29) | 9.1 (+0.45) | 4.3 (-0.42) | 2.1 (-0.04) |

- The shaped part as a whole moves with winning (correlation 0.95 over battles): gem_diff and countdown are
  potential differences and carry most of the signal long before the battle ends (with gamma 0.995 the final win is
  worth ~5% of its value 580 decisions earlier).
- The fight pieces (damage, kills, deaths, damage taken) roughly cancel out (+0.78 vs -0.75 per battle) and correlate
  only 0.58 with winning; damage still pays +0.45 in lost battles. They were raised 10x in auto15 so the agent would
  discover shooting at all; that is learned now.
- Conclusion: the reward is not obviously broken, so shaping changes will be small, careful gains, not a jump.
  Ideas, one per short run, each against the same baseline eval:
  1. Lower the fight weights back toward win/gems (`--dmg-reward 0.02 --kill-reward 0.05`): less incentive to trade
     shots while losing the gem race.
  2. Team-shared fight reward for the mixed team (each brawler gets the team's mean damage/kills instead of its own;
     host-side in train_ppo.py): rewards Colt and Dynamike for covering Shelly, not for stealing kills.
  3. A reward for supers that hit, only if policy_stats shows the super is still used at random.
