# Headless Driver Plan — libg.so offline battle loop

Status: architecture LOCKED (2026-10-07). **RUNTIME = v36.218** (2026-10-07): the symbolized v36
libg.so loads headless in our own process (Arxan runtime inactive — dlopen never calls JNI_OnLoad; Logic
.text is plaintext), so all offsets below are used DIRECTLY — no v29 mapping needed. v29 kept as fallback.

**M1 DONE (2026-10-07)** — `harness/jni/driver.c` (`driver <libdir> <dataroot> m1`): 56/56 resources
loaded, isLoaded=1, getGlobals non-null, item counts match CSV rows. Actual recipe (differs from the
original sketch below — `CSVNode(String const&)` only makes an empty named node, it does not parse):
```
LogicDataTables::constructInstance()                       @0x102538
arr = LogicResources::createDataTableResourcesArray()      @0x3a3f4c  // ArrayList{data,cap,size}, 56 entries
for i: res = arr->data[i]; name = String@res+8 (e.g. "csv_logic/globals.csv"), idx @+0x18, type @+0x1c
    read <assets>/<name>; split like DataLoaderCSV::load @0x5d1bdc ('\n', drop '\r', skip empty)
    -> one StringView(char const*,int) @0x4f645c per line (20 B each, points into buffer: keep it alive)
    node = new CSVNode (84 B); CSVNode(LogicArrayList<StringView>*, String const& name) @0x2577f4
    LogicResources::load(arr, i, node, NULL, false)        @0x2802c4  // i==0 resets; last i runs createReferences
```
Needs csv_logic/ + csv_client/ + localization/ from the APK assets (client tables are in the manifest).

**M2 DONE (2026-10-07)** — `driver ... m2 [loc= hero= gt= map= pm= diff= seed= team=]`. Replicates the
createOfflineServer Logic subset with explicit values instead of the `BattleMode::sm_offline*` statics the
client UI normally sets. Defaults = the game's **offline-practice** path (`HomePage::startGame` @0x511000):
`gameType = LBMS::getGameModeForOfflinePractiseFromVariation(location's variation)` @0x80af2c
(>2 teams -> 6, coop -> 7, Training(13) -> 8, else 1 = 3v3), `mapType=1`, `practiseMode=3`.
Tutorial preset (`MessageManager::startTutorial` @0x5a123c): `loc=Tutorial gt=5 map=4 pm=1`, hero ShotgunGirl.
`getPlayerCountWithGameMode`: 0,6 -> 10 · 3,7 -> 3 · 8 -> 1 · else 6. NULL heroUpgrades/emotes/skin accepted
by addPlayer (tutorial does the same). The `rand()` seed for setIteratedRandomSeed is replaced by our seed
-> same seed = identical bot lineup (verified). Verified: Gemgrab1 (6 players + OrbSpawner), Ball1
(6 + ball), Survival1 (10 + 19 boxes), Tutorial (6 + 12), all tick=0.
Server layout used: +0 GameObjectManager* (its +0 = ArrayList<GameObject*>), +0x3c bossData, +0x40
playerCount, +0x4c ArrayList<LogicPlayer*>; GameObject +8 LogicData*.

**M3 DONE (2026-10-07)** — `driver ... m3 [m2 args] [ticks=N every=N trap=0xOFF]`. Per-tick order copied from
the client's offline loop `BattleMode::update` @0x8c6e90:
```
LogicBattleModeServer::preTick()                    @0x3afb10  // = manager->preTick(players): REMOVES dead objects
LogicBattleModeServer::executeOfflineInputMessages() @0x27b4dc // offline player input queue (for actions later)
LogicBattleModeServer::executeOneTick(NULL, NULL)   @0x8a26d8  // what tickUpdate(1) @0x1304b0 does
stop when isGameOverSet() @0x5a3388
```
Skipping preTick = dead heroes never removed/respawned, area effects never expire (their encode overflows
7 bits -> "[ERROR] Write to BitStream out of range"), objects pile up, 6.5x slower.
**Vision-stream leak:** offline servers (+0xb4=1) encode a new 1.5 KB BitStream per tick and pass it to
`server+0x18 -> sendOfflineVisionUpdate(tick, bs)` (vtable slot 0xfc); the default `LogicGameListener`
version (@0x995174) is `bx lr` -> ~1.8 KB/tick leak. Fix in driver: copy `_ZTV17LogicGameListener`
(@0xc1421c, 356 B), point slot 0xfc at our fn (BitStream::destruct @0x9063b4 + free), store at +0x18.
Clearing +0xb4 instead CRASHES (online branch writes per-player streams from the inputs list; +0x44 = 1).
RSS now flat after ~600 KB warm-up. (That stream is the client's render feed — possible viewer source.)
**Obs fields (LogicCharacterServer):** +0x28 player index, +0x2c team (0/1), +0x90 HP, +0x94 max HP;
x/y via getX/getY. Characters list = manager+24 (getCharacters). Score: getTeamScore(team,false) @0x513c88.
**Results:** deterministic per seed. Gemgrab1 seeds 1-3 end at 2051-2765 ticks (different winners);
Ball1 ends at 4320; Survival1 and Tutorial stop at a 16000-tick cap with a passive player.
**Speed: ~15-18k ticks/s on ONE SDM710 core** (0.057 ms/tick) ≈ 850x real time; one Gem Grab match ≈ 0.15 s.
**Viewer (2026-10-07):** `python3 tools/replay.py [loc=.. seed=.. hero=..]` runs a battle on the tablet with
`rec=`, pulls `scratch/replays/<loc>_s<seed>.json`, embeds it into `tools/viewer/viewer.html` -> `.html`, opens
the browser (`--file x.json` rebuilds only; `#t=<tick>` deep-links). Recorder in driver.c: per tick all objects
`[gid,type,name,x,y,z,team,pidx,hp,maxhp]` (type via vtable slot 0x20: 0 char,1 proj,2 area effect,3 item),
tile grid re-sent when any tile changes. LogicTileMap = server+0x48: +0 LogicTile**, +0x68 w, +0x6c h;
LogicTile +0 = LogicTileData* (destroyTile -> Open). 300 logic units per tile. ~950 KB per Gem Grab match.
`trap=0xOFF` plants `udf` in our own process; SIGSEGV/SIGILL handler walks the r11 frame chain (libg keeps
APCS frames) — symbolize with v36_syms.tsv. Used to find the area-effect encode overflow.

**PLAYER ACTIONS DONE (2026-10-07).** Input path: `executeOfflineInputMessages` pops queued ClientInputMessages
(server +0x178 msgs / +0x184 player idx; msg +0x64 = ArrayList<ClientInput*>) and calls
`LogicBattleModeServer::handleClientInput(ClientInput*, playerIndex)` @0x75c168 per input. The driver calls
handleClientInput directly, between preTick and executeOneTick. ClientInput (36 B, ctor @0x8353b8, wire format
decode @0x864100): +0 seq, +4 type, +8 x, +0xc y, +0x10 target gid, +0x14 autoshoot, +0x15 has_target,
+0x18 pin id, +0x1c ?, +0x20 "carrying" (must match the hero's carry state for attack/super).
Types: **0 attack, 1 super** (x,y = offset from hero for straight shots; absolute ground point for throwers
`LogicSkillData::isPositionalTargeted` @0x965288 and for autoshoot; autoshoot+has_target -> server re-aims at
target->getX/YForAutoshoot; aim clamped to the visible rect and >= 100 units) · **2 move** (absolute target, server
clamps to 900 units = 3 tiles; client sends targets clamped to [1, w*300-2]) · **3 stop** · 5/6 super-aim UI on/off
· 7 carryable aim · **8 gadget** (player +0xd0 accessory) · 9 pin/emote · 10 steer controlled projectile · 12
showdown give-up. Ignored while server+0x73 is set (except 9), while server+0xe4 > 0, or if the hero is dead.
Client reference: BattleScreen::tryToActivateSkill @0x962d28, updateMovement @0x743080.
**Start countdown:** in 3v3 modes nobody moves for the first 120 ticks (6 s); serve `reset skip=120` skips it.
**Own-hero state:** LogicPlayer +0x38 super charge / +0x3c needed (hasUlti = equal), +0x10 hero gid; weapon
LogicSkillServer (getSkill(getWeaponSkill) @0x86ab74) +0xc ammo in 1/1000 shots, getMaxCharge @0x672bd0.
skills.csv CastingRange is in 1/3 tiles -> range = value*100 logic units (Shelly 23 -> 2300).
**Agent action** (driver `Action`): move dir (len<0.3 = stop, else move target = hero + 900*dir), attack/super
0 none / 1 auto-aim (closest living enemy in range passing isTargetVisibleToAttack @0x8db424; none in range ->
fire where the hero faces) / 2 aimed (dir, length = fraction of range), gadget. Presses on the first tick of a
step, steering re-sent every tick of the step.
**Teardown for reset:** `LBMS::destruct()` @0x57b170 (nulls the listener at +0x18, frees players/manager/tilemap)
-> `~LBMS()` @0x97cfc4 -> free. 40 full episodes in one process: RSS flat at ~48 MB.
**serve mode** (`driver <libdir> <assets> serve`): stdin `reset k=v..` / `step <repeat> mx my atk ax ay sup sx sy
gadget` / `quit`; one JSON line per reply `{"frame":..,"names":[[i,n]..new..],"done":0|1}` (+meta on reset: w, h,
players, range=[attack, super]). Frame = recorder frame + `own` [ammo/1000, slots, super, needed, hero gid] +
`inputs` [[type, world x, world y, target gid]] sent during the step. Table-load chatter goes to stderr.
Python side: `tools/bsenv.py` BrawlEnv/Action/Obs — **2.0 ms round trip per step over adb**; 437 steps/s (4
ticks each) incl. episode resets. Bottleneck is adb + JSON, not the sim (15k ticks/s): the RL wrapper should batch
several battles per driver process and use a compact binary obs; full frames only for watched envs.
**Watching / eval (off by default, zero cost when off):** `BrawlEnv(record=True)` + `save_replay()` -> replay.py
`--file`; `BrawlEnv(watch=True[, pace=1.0])` starts `tools/watch.py` Spectator (http://127.0.0.1:8766): viewer
follows the newest episode live (~0.5 s behind; skips ahead if the agent runs faster), episode picker, scrub;
`env.step(a, note={...})` shows agent numbers in the viewer. Viewer draws the inputs (move arrow, attack/super aim
line, auto-aim target ring), ammo/super HUD, plays by tick (frames may be k ticks apart).
`tools/play.py`: play yourself in the browser (WASD/arrows, Space/Enter/J auto attack, click aimed attack, E/K
super, right-click aimed super, Q gadget, P pause, R next seed); ~10 ms/step, saves `scratch/replays/play_*.json`.
`tools/example_agent.py`: heuristic agent template showing the inference + spectator loop.
`m3 act=script [think=4]` runs the same chase-and-shoot heuristic in C (Gem Grab seed 1: 12-2 vs passive 10-4).
**Known issue:** Showdown (Survival1) spawns our hero at (-150,150), outside the map — it never moves (this is
why Showdown ran to the 16000-tick cap). Needs a look at how solo-mode spawn slots / teams are assigned.

**RL WRAPPER STEP 1 DONE (2026-10-07): batched binary `vec` mode.** `driver <libdir> <assets> vec [n=16] [repeat=4]
[seed=1] [skip=120] [maxticks=12000] [vision=0] [cpus=<hex>] [loc= hero= ...]` runs n battles in one process behind a
binary stdin/stdout protocol (spec in the comment above `cmd_vec` in driver.c; client `tools/bsvec.py` VecDriver).
One round trip per step for all n battles: int32 request header + float32 actions[n][10] in; int32 header {magic, n,
busy_us, reset_us} + float32 reward[n] + int32 info[n][6] {flags (1 terminated, 2 truncated), tick, own score, enemy
score, seed, result} + float32 obs[n][35] out. Finished battles are rebuilt at once (seed += n), their row carries the
final info and the next battle's first obs. stdout is the protocol only (fd 1 is pointed at stderr before loading).
Obs v1 is a minimal placeholder (own state + 5 other heroes relative); reward = +1/-1/0 at game over (placeholder).
- **Isolation verified** (`driver ... isotest`): per-tick hashes of all objects of 8 battles run solo vs. in lockstep
  vs. staggered with a decoy battle on another map built/destroyed in between: identical. libg keeps no cross-battle
  state that matters, so many battles per process are safe.
- **Speedups found by simpleperf** (`simpleperf record -g` on the device works; libg symbols resolve):
  1. ~35% of a tick was the per-tick vision BitStream encode (`LogicGameObjectManagerServer::encode`) that only feeds
     the client. vec mode (`vision=0`, default) patches our in-memory copy of libg: executeOneTick+0x90 (0x8a2768,
     `ldrb r0,[r4,#0xb4]`) -> `b 0x8a2af4` (the LogicTime::increaseTick call), skipping both encode branches.
     `isotest novision=1`: Gem Grab (8 seeds, full games) and Ball1 (4 seeds) tick-identical with/without it.
  2. ~17% was A* pathfinding from re-sending the move input every tick. One move input walks the hero 900 units
     (25 ticks) along a path computed once, so steps now steer on their first tick and every STEER_TICKS=16 ticks
     (serve mode too, so watch/eval behaves the same as training).
- **Correctness** (`python3 tools/test_vec.py`): vec battle == serve-mode battle (unpatched, JSON) for the same seed and
  random actions — every hero identical every step, same game-over step and score (6 seeds, 3,800 steps); same seeds +
  actions -> identical obs; battle i of n=4 == the same seed alone; auto-reset info/seed stream; RSS flat over 100
  episodes; unknown map -> driver error message.
- **Speed** (random actions, Gem Grab, `tools/bench_vec.py`): on-device (`bench=` arg, no USB) a big core does
  24k ticks/s with n=1, 15k with n=16 (cache pressure grows with n); all 8 cores ~78k ticks/s (big ~15k, little ~8.4k
  each). End-to-end over USB, 8 drivers x 32 battles stepped async (each driver gets new actions as soon as it
  replies): **15.5k decisions/s = 62k ticks/s at repeat=4** (repeat 2: 23k/s = 47k ticks/s; repeat 8: 9.5k/s = 76k
  ticks/s ~ device ceiling); ~17 Gem Grab games/s. Lockstep over 8 drivers is ~half that (waits for the little
  cores). Single driver n=32: 3.2k decisions/s. Old JSON serve: 437/s. 8 drivers start in 1.5 s (`connect=False`
  then `connect()` loads tables in parallel).
- **Device notes:** cores 0-5 little (1.7 GHz), 6-7 big (2.0 GHz); governor schedutil. The vendor power management
  isolates idle cores at times (`/sys/devices/system/cpu/isolated`, e.g. "4"), so `taskset`/pinning to one core can
  fail with EINVAL; `cpus=` pinning is best effort (warns). Unpinned async drivers reach the numbers above.

**RL WRAPPER STEP 2 DONE (2026-10-07): observation v2** (protocol VEC_VERSION 2). Per battle a float32 vector (223) + a
uint8 tile grid (21x21), egocentric. The driver sends the layout string after the hello (`block:count:fields;...;
grid:21:flags`); `tools/bsvec.py` parses it: `obs = {"vec": [n,223], "grid": [n,21,21]}`, `d.view(vec, "enemy")["dx"]`,
`d.field_names()`, `d.describe(i)` (text dump + ASCII map). Blocks (field lists in OBS_BLOCKS in driver.c):
global(9) time, scores, both gem countdowns, seen items, gem mine dx/dy · self(19) alive, x/W, y/H, velocity, hp,
max hp, ammo, ammo slots, super, super ready, gems, respawn, ranges, facing, speed, hero id · ally x2 / enemy x3 (15)
present, alive, visible, dx, dy, seen_age, velocity, hp, max hp, gems, respawn, range, speed, hero id · item x8
(nearest, dx, dy, z) · projectile x12 (dx, dy, target dx/dy, enemy) · area x4 (dx, dy, enemy). Grid bits: 1 blocks
movement, 2 blocks shots, 4 bush, 8 destructible, 16 outside the map, 32 special (damage/heal/speed tiles), from
csv_logic/tiles.csv by tile name.
- **Fog of war = the game's own vision.** `LogicCharacterServer::isTargetVisibleToAttack(viewer, target, true)`
  @0x8db424 = "visible to the viewer's team" (no line-of-sight test; with `false` it adds LOS — that is what auto-aim
  uses): visible if the target attacked/was hit in the last 20 ticks (+0x19c/+0x68/+0x1a0), or
  `LogicGameObjectManagerServer::isPositionVisibleToPlayerTeam(mgr=viewer+0xc, tilemap, vx, vy,
  getTeamMemberPositions(server, team) @0x2253a4 (= server+0x7c[team], no copy), tx, ty,
  isPlayFieldMirroredForPlayer(team, server+0xa4) @0x8be73c)` @0xcf5bc: inside some team member's visibility rect
  (`isInsideVisibilityRect`; VISIBILITY_RAD_SUBTILES X 43 / Y up 33 / down 28 sub-tiles = ~+-14 x +11/-9 tiles) and
  `canPlayerSeeFromToNoLOS` (bushes hide unless close). Hidden enemies keep last seen pos/hp/gems; items/projectiles/
  area effects are filtered with the same position check; the gem mine is always known. Allies are always visible.
- **New fields found:** LogicPlayer +0x20 = score (gems carried in Gem Grab; team score = sum), +0x28 = death tick + 5,
  written ~5 ticks after death; hero respawns at +0x28 + 20*`LogicGameModeUtil::getRespawnSeconds(variation)` @0x8c6bd8
  (100 ticks in Gem Grab; tickSpawnHeroes @0x8f6368 starts the 40-tick spawn effect before). server+0xa4 = mode
  variation (0 Gem Grab), `LogicGameModeUtil::getBattleTicks(variation)` @0x3c6b14 = tick limit (Gem Grab 4320).
  Gem Grab win countdown (calculateIsGameOver @0x8c9484): server+0x98 end tick (0 = none), +0xac gems needed (10), +0xb0
  countdown seconds, +0x9c/+0xa0 team gem sums; re-checked at the start of the next tick (so the obs shows it only while
  the leader still has >= needed). getTeamScore @0x513c88: server+0xc8 == -1 -> sum of player+0x20 per team.
  LogicProjectileServer +0x44/+0x48 start x/y, +0x4c/+0x50 target x/y; LogicCharacterData +0xb0 speed
  (getCharacterData @0x108ea8); `LogicData::getInstanceID` @0x509ec8 (hero id = instance + 1). Object +0x2c team,
  +0x24 z. LogicCharacterServer vtable slot 0x30 = isAlive. Gem Grab objects: OrbSpawner (gem mine, item), Point
  (gem, item), HeroSpawn (area effect); summons (8-Bit's DamageBooster, ...) carry their owner's player index.
- **Verified** (`python3 tools/test_vec.py`, all pass): every obs field recomputed from serve-mode JSON frames of the
  same battle (unpatched game) matches every step — scores, time, mine, own state, velocities, all heroes, team gems =
  score, last-seen positions, respawn timers (~1 at death, <= 0.03 before respawning), items/projectiles/areas (real
  objects of that kind, nearest first, z, enemy flag), full 21x21 tile grid — on Gem Grab seeds 1,3,5 (+8,13,21) and
  Brawl Ball. Fog: hidden enemies are always in a bush or far from our team, never "in the open nearby".
  `driver ... deathtest` (prints LogicPlayer fields at deaths/respawns) and `movetest` remain as debug commands.
- **Speed:** obs v2 costs ~7% on the device (22.3k ticks/s n=1 on a big core). Replies grew to ~1.4 KB per battle,
  and a single-threaded host loop then waits on interleaved replies (recv 3.5 ms per 43 KB reply with 4-8 drivers;
  the link itself does 203 MB/s with 8 parallel streams). **One host I/O thread per driver** fixes it:
  8 drivers x 32 battles = **12.0k decisions/s = 48k ticks/s at repeat 4** (8.1k/s = 65k ticks/s at repeat 8),
  `tools/bench_vec.py --n 32 --drivers 8 --threads`. Still ~8 ms link overhead per 43 KB batch: float16 obs or sending
  the static map once (host-side crop) would cut the payload ~40-65% if needed.

**RL WRAPPER STEP 3 DONE (2026-10-09): reward pieces** (protocol VEC_VERSION 3). The driver reports 9 separate pieces per
battle-step (float32 rc[n][9] after reward[n]; names sent after the layout string); the host weights them
(`bsvec.REWARD_WEIGHTS`, `VecDriver(weights=...)` / `d.set_weights()`, no driver restart), `d.parts` holds the raw
pieces. reward[n] from the driver is the "win" piece only. Serve mode sends the same pieces as `"rc"` per reply
(`bsenv.Obs.reward_parts`, `obs.reward(weights)`). Code: `rew_read`/`rew_step` in driver.c (shared by both modes).
- Pieces (own brawler/team, natural signs, weights carry the sign; defaults in brackets):
  win +-1/0 at game over [1.0] · gem_diff change of (own - enemy team gems) [0.1] · countdown change of the
  win-countdown potential (+progress 0->1 while ours runs, -progress while theirs runs; Gem Grab only) [0.5] ·
  gems_carried change of own carried gems [0.05] · kill [0.05] · death [-0.1] · dmg_dealt own damage to enemy
  brawlers / 1000 [0.005] · dmg_taken own hp lost in lives, low-hp weighted: F(h0) - F(h1), F(h) = 1.5h - h^2/2
  (0.5/life at full hp, 1.5/life near 0, full -> dead = 1; healing not rewarded) [-0.1] · low_hp seconds alive below
  10% hp (sampled at step end) [-0.02].
- gem_diff and countdown are potential differences: their episode sums are the final gem difference and +-1 when a
  countdown ended the game, so they guide without changing what winning means. kill/death/damage are not; their
  weights are kept small (Shelly deals 10-45k damage per game, so dmg_dealt at 0.02 rivalled the win).
- **New fields:** LogicPlayer battle stats live in a stat list (+0x54 pointers to {id, value}, +0x5c count); getters
  getKills @0x506a80 (stat 2), getDamageDoneToBrawlers @0x1422a0 (3), getDamageReceived @0x66e324 (9), getDeaths
  @0x90cbd4 (13), getGemsCollected @0x45ffe0 (16), getGemsLost @0x90dad4 (17), getHealingDoneToBrawlers @0x4a146c
  (11), getHealingDoneToTeamMates @0x23164c (25), getHealingDoneToSelf @0x8337f8 (26).
- **Verified** (`tools/test_vec.py`): vec pieces == serve pieces every step; each piece agrees with the frames (score
  and carried-gem changes, own deaths, hp lost with the weighting, low-hp seconds, win at game over; kills only on
  steps where an enemy died, damage only where an enemy's hp changed: regen can outweigh the hit within a step); episode sums of gem_diff/countdown as above.

**RL WRAPPER STEP 4 DONE (2026-10-09): `BrawlVecEnv`** (`tools/bsvecenv.py`, numpy only, no driver change). D driver
processes x n battles behind one gymnasium-style vector env; one persistent host I/O thread per driver; replies are read
straight into env-wide arrays (`VecDriver(out=...)` + `recv_batch()`, no copies or concatenation).
- **Lockstep is slow, use async.** Stepping all 8 drivers together and waiting for all of them: 63 ms median per step
  although each driver simulates in ~16 ms. Every step one random driver's reply arrives ~40 ms late through USB/adb
  (arrival times sorted: 18 20 22 24 26 29 36 59 ms; which driver is last changes). Plain VecDrivers in lockstep are
  just as slow, so it is the link, not the env. Fix: EnvPool-style `send(actions, env_id)` / `recv()`; `recv()`
  returns every group that has answered (a group = one driver by default) and the policy acts on just those rows.
- **Speed** (repeat 4, random actions, fake policy that sleeps per call): 8x32 12.2k decisions/s (97% of
  bench_vec --threads), 8x64 14.2k, **8x96 15.8k = 63k ticks/s (~80% of the tablet ceiling), 15.3k / 14.7k with a
  2 / 5 ms policy**, 8x128 15.4k; 16x32 12.8k, 24x16 8.5k (more drivers than cores does not help). Lockstep step():
  4.4k. Default is 8 x 96 = 768 battles. Driver RSS 72 MB at n=128.
- **info:** env_id, parts [rows, 9], the INFO_FIELDS columns, final, episode_return / episode_length (finished rows),
  crashed. reward = parts @ weights on the host (`env.set_weights()` at any time).
- **Seeds:** driver k starts at seed + k*10^7 (training default seed 10^7, eval at 10^6 + k never collides). A restarted
  driver continues above every seed it used (a crashing battle is not retried).
- **Crash/hang:** a driver that dies or misses `timeout` (30 s) is killed, also on the tablet (the driver's shell
  prints its pid, then exec's the driver: `VecDriver.remote_pid`, `kill()`), and restarted (0.8 s); its rows come back
  truncated + crashed with reward 0 and the new battles' first obs; others keep stepping. > max_restarts (8) in 10 min
  -> RuntimeError. Truncation otherwise only at maxticks 12000 (Gem Grab ends by 4320), so no final obs is sent:
  bootstrap truncated rows from the previous step's value.
- **Verified** (`tools/test_vecenv.py`, ~90 s, ALL PASS): env == plain VecDrivers bit for bit (lockstep, single groups
  out of order, async with groups at different paces); episode stats; kill -9 and SIGSTOP of a driver (restart, seeds,
  other driver unaffected, matches its reference); restart limit; no driver process left after close().

**RL WRAPPER STEP 5 DONE (2026-10-09): cores / speed.** Result: keep 8 unpinned drivers x 96 battles; add
pipelining (protocol **VEC_VERSION 4**: STEP = {MAGIC, STEP, repeat, count} + int32 start + act[count][10] steps
battles start..start+count-1, the reply covers that range; writev of the section slices). BrawlVecEnv(pipeline=2)
keeps both halves of a driver in flight.
- Steady state ~16k decisions/s = ~65k ticks/s = **~30 games/s** (random play, after a 70 s warm-up; all battles start
  together so short runs see no game ends). Tablet ceiling without any USB traffic (8 bench drivers at once):
  67-74k ticks/s for n = 8..96, so the env is at ~93% of the hardware. Per process: drivers 71-91% CPU, adbd 28%.
- Tried, not better: pinning one driver per core (15.7k), 16 drivers x 48/64/96 (14.9-15.4k), 12 x 64 (14.1k),
  pipeline 3/4 (15.8k/15.5k). Pipelining gains only when the policy takes time: 5 ms policy 15.1k -> 15.4k,
  10 ms 13.3k -> 14.5k.
- Profile (simpleperf, one driver): game tick ~75% (bots' A* pathfinding ~33%), our handleClientInput ~8% (our hero's
  move = A* to a point 900 units ahead every step), vec_obs ~5%. Per tick cost rises with battles per driver (one
  core alone: n=1 25.8k ticks/s, n=32 18.9k, n=96 17.6k: cache), but with 8 drivers running the shared cache/memory
  is the limit for any n.
- USB round trip with 1 battle: 1.1 ms median (dock, USB 3 bus): the dock is not a bottleneck.
- Verified: test_vecenv.py ALL PASS incl. a crash while the host holds one half (that half reports truncated +
  crashed with its old seeds on its next step). bsvec smoke test reproduces the same battles as protocol v3.
  test_vec.py (vec vs serve, 15 min) not rerun: the step loop only changed its row range.

**RL WRAPPER STEPS 6/7 DONE (2026-10-09): watching + soak.**
- Step 6: `episodes.EnvRecorder(env, env_ids, out_dir, keep, extras)` attaches to `env.on_step`, logs the chosen
  battles from their first step (actions + reward + every reward piece + extras such as the value estimate); crashed
  episodes are dropped. Recorded games re-simulate exactly in serve mode (same final score + tick), checked by
  test_vecenv.py "watch". Nothing attached = no cost.
- Step 7: `tools/soak_vecenv.py --minutes 20` (training config, 8 x 96, recorder on battles 0 and 767). Result:
  speed 16.1k -> 15.8k decisions/s (x0.98), 28-30 games/s, 0 crashes, driver memory 529 -> 530 MB total
  (66 MB max per driver), hottest sensor 75 C flat, 20 episodes recorded: SOAK PASS. test_vecenv.py ALL PASS
  afterwards (179 s).

**RL STEP 9 DONE (2026-10-09): steer the whole own team** (protocol VEC_VERSION 5). Bot control in v36:
`LogicCharacterServer::tickAI` (0x81a124) calls `tickBot` (0x487c0c) only if hero byte +0x125 is set (and the hero is
no boss, server+0x2ae == 0). `LogicBattleModeServer::setControlledByBot(int player, bool)` (0x6aec9c) writes player
+0x31 (and zeroes +0x7c when turning a bot on) and the hero's +0x125; `spawnHero(data, upgrades, team?, ?, bool bot)`
stores its last argument at hero+0x125 and `tickSpawnHeroes` passes player+0x31, so respawned heroes stay ours.
`handleClientInput(input, playerIndex)` (0x75c168) has no bot check: inputs for any player index are applied.
Driver: build_battle key `agents=k` calls setControlledByBot(p, 0) for player 0's first k-1 teammates (player order;
`team_agents()` gives the same list); `sim_tick_agents()` applies one Action per agent before the tick; `Env` holds
per-agent `Agent {pidx, ctl, lx, ly, rew}`; `vec_obs(e, j)` builds agent j's egocentric view (its allies = the other
team members in player order), `vec_remember()` updates the velocity memory once per battle-step; reward pieces per
agent (`rew_read/rew_step(..., pidx)`). Vec rows = battles x k, steps must cover whole battles. Serve: `step` takes 9
numbers per agent. Checks: test_vec.py ALL PASS (agents=1 unchanged), tools/test_agents.py ALL PASS.

## The breakthrough: the game hands us the recipe

`BattleMode::createOfflineServer(LogicHeroUpgrades*, LogicBattleEmotes*, LogicIntArrayList* eventMods, int raidDiff, int gameObjArg)`
@ v36 **0x3463ec** is the game's OWN headless offline-battle factory. We do not guess an init
order — we replicate its Logic-layer subset. Annotated disassembly (v36):

```
server = operator_new(744)                                  // sizeof(LogicBattleModeServer)=744
gm     = <global current game mode id>
pc     = LogicGamePlayUtil::getPlayerCountWithGameMode(gm)  @0x8ed0a4
LogicBattleModeServer::LogicBattleModeServer(server, gm, pc)@0x3bf880   // ctor(int gameMode,int playerCount)
server[0xc4] = <global byte> ; server[0xb4] = 1
LogicBattleModeServer::setRandomSeed(server, seed)          @0x5d8398
LogicBattleModeServer::setLocation(server, LogicLocationData*) @0x3aabe8
server[0x3c] = LogicDataTables::getTable(22)->vtbl[0x14](0) @getTable 0x4f2458
LogicGameObjectManagerServer::setLogicBattleModeServer(server[0], server) @0x8c4640
LogicBattleModeServer::generateTileMap(server, pc, w, h)    @0x84d504
  (offline branch) LogicRandom::setIteratedRandomSeed(...)  @0x1f0388 via getLogicRandom @0x31b4c0
LogicLong id; LogicLong::LogicLong(&id)                      @0x4c4f2c   // player id 0
dd = LogicDataTables::createDefaultDisplayData(name)         @0x86e5b8   // name via StringTable::getString @0x80a8cc
LogicBattleModeServer::addPlayer(server,&id,charData,heroUpg,emotes,team,dd,flag,skin) @0x70b558
server[0xb7] = (gm-1 < 2)
LogicBattleModeServer::setBotDifficulty(server, diff)        @0x292104   // leaf: str r1,[r0,#0xb8]
LogicBattleModeServer::setRaidDifficulty(server, raidDiff)   @0x6037cc
LogicBattleModeServer::addGameObjects(server,1,true,<charDataList>,gameObjArg,0) @0x3274e4  // spawns bots/objects
  if(<flags>){ server[0xbe]=1; setBotDifficulty(server,4); }
for e in eventMods: LogicBattleModeServer::addEventModifier(server,e) @0x39edec
// then loop: executeOneTick(inputList, outStream)           @0x8a26d8
```

Note: the real `createOfflineServer` needs a constructed `BattleMode* this` (client wrapper) and many
loaded globals. We DO NOT call it directly — we call its **Logic-layer subset** above, bypassing the
client wrapper and the player-avatar branch (taking the offline/training branch that builds display data
from a StringTable name instead of a client avatar).

## Prerequisite: data tables loaded (the linchpin)

Confirmed feasible OFFLINE and in pure Logic layer:
- All 33 `csv_logic/*.csv` are **plaintext ASCII/UTF-8, uncompressed** in the authentic v29 APK
  (`apk/extracted-29/assets/csv_logic/`). No SC/LZMA decompression needed.
- Logic-layer load boundary takes an already-parsed `CSVNode`, so we never touch the engine's
  async file/asset/JNI/task-queue machinery.

Storage model (confirmed from disasm): `getTable(i)` @0x4f2458 returns `g_tableArray[i]` (global word
array of CSVTable*). `getGlobals()` @0x6fd8cc returns a CACHED `LogicGlobals*` at singleton+0xc, built
during `createReferences()`. So the numeric index per table is baked into the game's resource manifest
and consumed by createReferences — we do NOT reconstruct it by hand.

**Chosen M1 route — reuse the game's manifest (minimal RE, robust):**
```
LogicDataTables::constructInstance()                   @0x102538  // singleton + table array
arr = LogicResources::createDataTableResourcesArray()  @0x3a3f4c  // 56x LogicDataTableResource(name,idx,type)
for i in 0 .. arr.size:                                           // ArrayList = {ptr,size,cap}; read fields direct
    name = arr[i]->getFileName()                       @0x7cfa0c  // runtime-DECRYPTED -> e.g. "globals.csv"
    buf  = fopen(csv_logic/<name>)                                // our C; files are plaintext on disk
    String s(buf)                                                 // String ctor/assign
    CSVNode node(s)                                     @0x573754  // CSVNode::CSVNode(String const&)
    LogicResources::load(arr, i, &node, NULL, false)   @0x2802c4  // uses resource[i] index/type internally
LogicDataTables::createReferences()                    @0x73aae0  // builds LogicGlobals etc. from tables
assert LogicDataTables::isLoaded() @0x27b644 == true  &&  getGlobals() @0x6fd8cc != NULL (known field reads right)
```
Why robust: CSV filename literals are NOT plaintext in libg.so (encrypted at rest, decoded on use), but
`getFileName()` returns the decrypted name at runtime — so matching to our on-disk files needs no static
string recovery, and per-table index/type come straight from the game's own resources array.
Fallbacks: raw `initDataTable(CSVNode*,int)` @0x5010f4 (+`initTilemap` @0x57ac98) with a hand-built index
map from createReferences immediates; or the full game loader `DataTableResourcesLoader::loadDataTables(bool)`
@0x829060 (higher file/asset dependency risk).

## Observation accessors (read state each tick) — v36 addrs
- getState @0x32a6d0 (+0x64) · getX @0x7fc5dc (+0x1c) · getY @0x7587ac (+0x20)
- getGlobalID @0x82c034 (+0x4) · getBotSkill @0xfee68 (+0xb8) · getTick @0x913168
- getTeamScore @0x513c88 · isGameOverOrRoundResetting @0x5039fc · getTicksGone @0x4f122c
- game-object enumeration: LogicGameObjectManagerServer (iterate server[0] manager list)

## executeOneTick signature
`executeOneTick(LogicArrayList<BitStream*>* inputs, BitStream* out)` v36 @0x8a26d8 / v29 @0x3198c4 (CONFIRMED).
For a no-input tick: pass empty list + a scratch BitStream. Bot AI runs inside tick() regardless of inputs.

## Milestones
- **M1 — tables loaded**: constructInstance -> load 33 CSVs -> createReferences -> isLoaded()==true,
  getGlobals() returns non-null and a known global reads correctly. THE linchpin.
- **M2 — server constructed**: replicate createOfflineServer Logic-subset -> non-null server,
  getTick()==0, player + bots present (addGameObjects ran).
- **M3 — ticking**: loop executeOneTick N times; getTick() advances; read obs (positions/state/score);
  reach a game-over. Measure ticks/sec headless.

## v29 mapping worklist (map each v36 addr above -> v29, then verify by call in hostprobe)
CONFIRMED v29: executeOneTick 0x3198c4, addGameObjects 0x17b938, tick 0x2bd008,
calculateIsGameOver 0x1c8bdc, calculateXpBonuses 0x5dc818.
PENDING v29 (priority order = M1 then M2):
  M1: constructInstance, CSVNode::CSVNode(String&), initDataTable, initTilemap, createReferences,
      isLoaded, getGlobals, getTable, String ctor/assign.
  M2: LogicBattleModeServer ctor(0x3bf880), getPlayerCountWithGameMode(0x8ed0a4), setRandomSeed(0x5d8398),
      setLocation(0x3aabe8), setLogicBattleModeServer(0x8c4640), generateTileMap(0x84d504),
      LogicLong ctor(0x4c4f2c), createDefaultDisplayData(0x86e5b8), StringTable::getString(0x80a8cc),
      addPlayer(0x70b558), setBotDifficulty(0x292104), setRaidDifficulty(0x6037cc), addEventModifier(0x39edec).
  obs: getState,getX,getY,getGlobalID,getBotSkill,getTick,getTeamScore,isGameOverOrRoundResetting.
Mapping method: rare-constant matcher (tools/match_v36_v29.py) where it fires; else dynamic
verification in hostprobe (call v29 candidate, check behavior) — user-chosen approach.
```

## BOT DIFFICULTY / BRAWLER KEYS (2026-10-10)
- All runs up to team45 trained and evaluated against bot difficulty 0 (driver default `diff=0`,
  `LBMS::setBotDifficulty` -> server+0xb8; bot skill = player+0xe0 if set, else that; read by `tickBot`,
  `aiUseOffensiveSkills`, `tickRapidFire`). team45 on 64 eval battles, all 3 steered: diff 0 92%, 1 84%, 2 38%, 3 20%,
  4 16%, 6 11%, 9 16% (bots stop getting stronger around 4; battles at diff >= 3 simulate ~1.8x slower).
  Shelly-only bots: diff 0 100%, 2 47%, 4 19%.
- What the difficulty changes (disassembly, 2026-10-10): `tickBot` @0x487cc0 sets the bot's think interval to 20
  ticks for skill < 3, 10 for skill 3, 6 for skill >= 4 (server+0xbe, the game's own hard-bot flag, forces 6 and
  setBotDifficulty(4)); `aiUseOffensiveSkills` @0x1c7458 skips its offensive-skill use with 70% chance at skill 0, 50% at
  1, never from 2 on. Nothing else reads it, so difficulty 4 = 5 = ... = 9: 4 is the strongest bot the game has.
- Driver key `bots=<Name>[,<Name>...]` (all modes): bots' brawlers drawn only from these internal names (characters.csv,
  e.g. ShotgunGirl = Shelly). addBots picks heroes with `LogicDataTables::getRandomHeroForBot` @0x108414, which skips
  every hero on the list given to addGameObjects; the driver lists all other heroes (table 16, isHero). Unknown names
  -> error. With agents=3 our teammates are bot picks too, so `bots=ShotgunGirl` = six Shelly. Previous driver build on
  the tablet: driver.v6.
- train_ppo: `--diff 2,2,3,3,4,4,4,4` (bot difficulty per driver, cycled), `--bots ShotgunGirl`, `--eval-diff`
  (default: highest of --diff; `--eval` uses it too, output dir eval_<ckpt>_diff4_botsShotgunGirl), `--minutes N`
  (stop after N minutes); log columns win_d<k> = training win rate per difficulty. The learning rate now decays from
  the resumed step (was from step 0: a resumed run started at a fraction of --lr).
- runs/hard20 (2026-10-10, --resume team45 --diff 2,2,3,3,4,4,4,4 --bots ShotgunGirl, 30M -> 40M decisions, 21 min,
  0 crashes, ~14k decisions/s): diff-4 evals (32 battles) 38 -> 41 -> 28 -> 50 -> 62 -> 44 -> 62 -> 72 -> 62 -> 69%.
  Final 128-battle evals: diff 4 Shelly-only **68.8% [60-76]** (team45 23.4%); diff 4 random brawlers 22.7%
  (team45 24.2%: the Shelly-only training does not carry over to mixed lineups); diff 0 random 93.8% (no forgetting).
- Driver key `mates=<Name>[,<Name>...]` (2026-10-10, 3v3 only): player 0's two teammates get these brawlers (cycled),
  added with addPlayer like player 0 *before* addGameObjects, which then only spawns the enemy bots (still random, or
  `bots=`). Teammates beyond agents=k are handed to the bot AI with setControlledByBot(p, 1) (checked: they walk and
  shoot). Unknown names -> error. Previous build on the tablet: driver.v7 (bots= only). train_ppo `--mates ShotgunGirl`.
  Intended setting from now on (user): our team always Shelly, enemies random. Baselines there, all 3 steered, diff 4,
  64 battles: team45 25.0%, hard20 31.2%. (`eval.py ... mates=ShotgunGirl`, agents=1: idle 6% — Shelly bots are weaker
  teammates than a random lineup.)
  Known: the first update after --resume jumps (kl ~0.3-0.5, ev negative), harmless so far.

## HANDOFF HISTORY: RL wrapper steps 1-9 + first training runs (2026-10-10)
- M1-M3: data tables, offline battle construction, tick loop (`harness/jni/driver.c`).
- Viewer / replays (`tools/replay.py`, `tools/viewer/viewer.html`), human play (`tools/play.py`), live spectator for
  agent inference (`tools/watch.py`, `BrawlEnv(watch=True)`), JSON env `tools/bsenv.py` (serve mode, for eval/watch).
- RL env wrapper, 7-step plan (approved by the user):
  1. **DONE** batched binary `vec` mode: N battles per driver process, auto-reset, in-memory skip of the per-tick vision
     encoding (verified tick-identical), steering once per step. Client `tools/bsvec.py` (VecDriver).
  2. **DONE** observation v2: 223-float egocentric vector + 21x21 tile grid, fog of war from the game's own vision
     check, verified field by field against the JSON frames (`tools/test_vec.py`, all pass).
  3. **DONE** reward pieces (protocol v3): 9 pieces per step from the driver (win, gem_diff, countdown,
     gems_carried, kill, death, dmg_dealt, dmg_taken with low-hp weighting, low_hp), weighted on the host
     (`bsvec.REWARD_WEIGHTS`, `VecDriver(weights=...)`, `d.parts`); eval reports return + mean piece sums.
     Details: docs/driver_plan.md "STEP 3".
  4. **DONE** `BrawlVecEnv` (`tools/bsvecenv.py`): 8 drivers x 96 battles, async EnvPool-style `send`/`recv`
     (lockstep is 3x slower: USB stragglers), 15.8k decisions/s = 63k ticks/s, crash/hang restart, per-driver seed
     blocks, episode stats in info. Tests `tools/test_vecenv.py` (~90 s). Details: docs/driver_plan.md "STEP 4".
  5. **DONE** speed: 8 unpinned drivers x 96 + pipelining (protocol v4, 2 halves per driver in flight) = ~16k
     decisions/s, ~65k ticks/s, ~30 games/s; ~93% of the tablet's own ceiling (pinning / more drivers were slower).
  6. **DONE** watching: `episodes.EnvRecorder(env, env_ids=[...], out_dir=...)` hooks into the env, logs the chosen
     battles' actions (+ reward pieces, + any per-step numbers such as the value); every finished game re-simulates
     exactly into a viewer replay (`tools/episodes.py file.npz`). Off = not attached = no cost.
  7. **DONE** `tools/soak_vecenv.py --minutes 20`: training config + recorder; prints speed, games/s, crashes, driver
     memory and tablet temperature per minute and judges the trends (result in docs/driver_plan.md "STEP 6/7").
  8. **STARTED** training: `tools/train_ppo.py` (PPO, CPU torch from the shared venv `../../.venv`, nothing
     installed). Run: `../../.venv/bin/python tools/train_ppo.py` -> runs/<date-time>/ (log.csv, last.pt, best.pt;
     `--watch` records battle 0, `--resume runs/X/last.pt`). Smoke run (2.5 min, 1.2M decisions): training win rate
     5% -> 20-28%, value explained variance 0 -> 0.7. ~8.5k decisions/s overall: collect 3.0 s + update 2.2-4 s per
     49k-sample rollout (the laptop's update, not the tablet, is the limit; a grid CNN was 2.5x slower than the
     linear grid layer; 8 torch threads ~3x slower than 4 = physical cores). Possible later gain: overlap the
     update with the next rollout.
     Watch: `python3 tools/dashboard.py` (http://localhost:8800; newest run's charts + every eval battle and
     --watch training game, click = re-simulate once on the tablet ~3 s, then the viewer opens). Separate process,
     only reads runs/<run>/ files: no cost to training. Evals default to 32 battles every 20 updates (~15 s / 2 min).
     First run `runs/first20` (2026-10-09, 10M decisions, 21 min, 16.5k games, 8.9k decisions/s): training win
     23% -> 38%, evals 19 -> 34 -> 47 -> 50 -> 31 -> 47 -> 41 -> 38 -> 47 -> 44% (32 battles each, noisy). Final model
     on 128 eval battles: 38.3% [30-47] wins, gem diff -1.27 (random 17%, gems/chase heuristics 72%). One driver
     crash (96 battles, auto-restarted; reason lost to a log filter, now saved to runs/<run>/crashes.jsonl).
     Measure any model: `train_ppo.py --eval runs/X/last.pt --eval-episodes 128`.
     first20 diagnosis (`tools/policy_stats.py runs/X/last.pt`: per-head entropy, when it attacks, aim error, ammo,
     time at the mine): every head still ~random, attack pressed 88% regardless of enemies, aimed shots at chance
     (13% within 23 deg vs 12% random), damage reward too small (0.005/1000 hp). Fix (now defaults): `--aim auto`
     (attack/super menu = nothing or the game's auto-aim; `--aim free` = old 10-entry menu), `--dmg-reward 0.05`,
     `--kill-reward 0.1`, `--waste-cost 0.01` (attack pressed with ammo but no visible enemy in range; log column
     waste%). Run `runs/auto15` (2026-10-09, 8M decisions, ~15 min): training win 43% -> 70%, waste% 31 -> 8,
     final 128-battle eval **74.2% [66-81], gem diff +6.3** (gems heuristic 72%). Behaviour: attacks 49% of steps
     with an enemy in range vs 23% without; empty clip 51% of alive time but only +3 points death risk (17% vs 14%
     die within 3 s with an enemy in range) -> no zero-ammo cost added. Super/gadget heads still ~random.
     Ideas raised: longer run / resume; control the whole own team (3 rows per battle, driver work), later
     self-play with team-1 mirroring + opponent pool; behaviour cloning of the gems heuristic now less needed.
  9. **DONE** whole own team steered (driver protocol v5, 2026-10-09): battle key `agents=k` (1..3) hands player 0's
     first k-1 teammates from the bot AI to us with the game's own `LogicBattleModeServer::setControlledByBot(p,
     false)` (0x6aec9c: player+0x31 and hero+0x125; `LogicCharacterServer::tickAI` runs `tickBot` only when hero+0x125
     is set, respawns copy player+0x31). Vec mode: n = rows = battles x k, row r = brawler r % k of battle r // k,
     each with its own egocentric observation (allies = the other team members in player order), action and reward
     pieces (win/gem_diff/countdown team-wide, the rest per brawler); step ranges must be whole battles. Serve mode:
     `step` takes 9 numbers per agent; episodes store [T, 10k] actions and re-simulate exactly. agents=1 is
     identical to v4 (test_vec.py ALL PASS). `tools/test_agents.py` (~2 min): teammates obey/stand idle, the three
     rows' views agree (4993 battle-steps), team pieces/info equal, 3-brawler replay exact. Previous driver build
     kept on the tablet as driver.v4. train_ppo default `--agents 3 --battles 32` (768 rows): ~13-14k decisions/s
     (collect 1.6 s instead of 3 s per rollout). New baselines with all 3 steered (eval.py ... agents=3, 64 battles):
     gems heuristic 26.6%, random 0% (bot teammates carried a lot: 72% / 17% with agents=1).
     First team run `runs/team15` (10M decisions, ~15 min, 0 crashes): training win 2% (first 2M) -> 34% (3M) ->
     54% (5M) -> 57% (8M) -> 63% (9M), still rising; 128-battle eval **58.6% [50-67], gem diff +4.3** with all three
     steered (gems heuristic team 27%); the same model as player 0 with two bot teammates: 56.2% (auto15: 74%).
     Continued as `runs/team45` (--resume runs/team15/last.pt --total-steps 30000000, 20M more decisions, ~30 min,
     0 crashes): evals 66 -> 75 -> 84 -> 91 -> 97% and then 88-97%; final 128-battle eval **94.5% [89-97], 121/7,
     gems 12.9 vs 0.9** with all three steered; as player 0 with two bot teammates 79.7% [72-86] (auto15: 74%).
     Dashboard note: it now re-simulates each replay in a fresh process (`tools/episodes.py`), because a dashboard
     started before the v5 change replayed 3-brawler battles with player 0's actions only (teammates idle, blue lost).
