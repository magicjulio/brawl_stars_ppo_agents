# 03 — Offline client via Frida gadget (option 1, community recipe)

Date: 2026-10-06. Status: **in progress — building gadget-injected client.**
Supersedes the static-patch approach in `02_guard_neutralization.md`.

## Decision

User chose **option 1: build on the community pre-patched v36 approach.** Both community repos
(ShadowwDev/BrawlStars-v36, erder00/royale-brawl-v29) are **offline server emulators**; each ships
its client as an APK with a **Frida *gadget* baked in** that applies the bypass at **runtime**, not a
static file patch. We replicate that *recipe* on our own verified-authentic v36.218 APK rather than
running an untrusted MEGA binary — an unknown modified client could do arbitrary networking, which
would violate the hard constraint **"never reach public Supercell servers, research only, fully offline."**

## Why this clears the Arxan wall (that `02` hit)

Arxan keeps `.text` mapped `r--` and runs **decrypted copies** of code in memory. A static byte-patch
to the on-disk `.text` never executes. A Frida `Interceptor.replace`/`.attach` installs a trampoline on
the **in-memory, post-decryption** copy (the v29 script forces it with `Memory.protect(addr,len,"rwx")`),
so the hook actually runs. It also lets us touch **only** the handful of functions that matter and leave
the `lv1_*` frida/memory-guard subsystem alone — that subsystem is entangled with Arxan's own
decrypt-on-fault handler, which is why stubbing it statically in `02` broke code decryption.

## The bypass recipe (from v29 `liberder.script.js`, ported to v36 by symbol name)

v29's entire anti-cheat bypass is **4 hooks** (not the 24 we static-patched):

| purpose | v36 symbol (resolve by name; libg.so is NOT stripped) | v36 addr | action |
|---|---|---|---|
| Arxan entry | `_Z20g_createGameInstancePii` | 0x33bcb8 | `Interceptor.replace` → jump to `GameMain::getInstance` |
| — target | `_ZN8GameMain11getInstanceEv` | 0x2897e4 | (replacement target) |
| antihack flags | `_ZN9AntiCheat16getAntihackFlagsEv` | 0x6b7cd0 | replace → return 0 |
| guard callback | `_ZN9AntiCheat14guard_callbackEi` | 0x8339f4 | replace → noop |
| signature check | `_ZN5titan3com9supercell5titan7GameApp16isSignatureValidEv` | 0x595c20 | attach, onLeave retval = 1 |

All ARM mode (even addr). Resolve at runtime via `Module.enumerateSymbols()` on the mangled names so
Thumb/ARM bit and base-relocation are handled for us. **Frida-detection is NOT bypassed in v29** — we
start there and only add runtime no-ops on `lv1_detect_frida_*` / `lv1_frida_killer` *if* the gadget
gets killed on v36 (TBD empirically).

## Networking + offline battle (from v29, structural offsets must be re-derived for v36)

- `connect()` hook: when `dest port == 9339`, rewrite sockaddr to `127.0.0.1:<local>` and capture fd.
- Pump the in-process message queue: hook `pthread_cond_signal` (send) and `select` (recv) — v29 reads
  `ServerConnection`/`Messaging` struct offsets; **these offsets differ in v36** and must be recovered
  from our symbols/DWARF before messaging works.
- `OfflineBattles()`: force the offline-battle check true, `LogicBattleModeServer::setBotDifficulty`
  (write difficulty at `this+184` in v29 — verify v36 offset), force game-mode arg.
- Local server: **ShadowwDev/BrawlStars-v36** Python server (matches v36.218), run on `127.0.0.1:9339`,
  fully offline. Only Gem Grab / Bounty / Big Game are reliable — fine for a first RL task.

## Observation-space hooks (our actual goal, once in a battle)

- `LogicBattleModeServer::tick()` 0x1ee63c, `executeOneTick(...)` 0x8a26d8 — per-tick entry to dump state.
- `LogicBattleModeServer::setBotDifficulty(int)` 0x292104.
- `LogicCharacterServer::tickBot()` 0x487c0c — bot AI per character.
- Struct layouts come from DWARF in our libg.so (no runtime needed for layout analysis).

## Build plan (gadget-injected client)

Target APK: `apk/brawl-stars-36-218.apk` (authentic, Supercell-signed). Package `com.supercell.brawlstars`,
launcher activity `com.supercell.brawlstars.GameApp`, **no custom Application class**.

1. Download frida-gadget for `android-arm` (armeabi-v7a) matching frida 17.x (GitHub frida/frida releases — trusted).
2. Add gadget as `lib/armeabi-v7a/libgadget.so` + `libgadget.config.so` (script mode, embeds our bypass JS).
3. Make the app load it early: inject `System.loadLibrary("gadget")` into `GameApp.onCreate` (baksmali the
   dex holding GameApp → edit → smali → swap dex back; avoids full apktool resource re-encode on a 183 MB game).
4. `zip -0` the modified dex + gadget libs into an APK copy, strip old signature, `zipalign -p 4`, debug-sign (v2+v3).
5. Install to offline AVD `bs_api30` (`--abi armeabi-v7a`), launch, watch logcat: expect gadget to load and run
   the bypass; confirm past the ~2.6 s guard crash and into the client. Iterate on frida-detection only if killed.

## Tooling status (2026-10-06)

Present: frida CLI 17.18.0, apktool 2.7.0, baksmali/smali 2.5.2, aapt2/zipalign/apksigner (SDK build-tools),
git, python3. Missing: frida-gadget `.so` (download), objection (not needed — manual smali inject),
frida-python module for sys python3 (CLI is enough).

## Hard constraints (unchanged)

Fully offline. Never contact public Supercell servers. Research only. Do not run untrusted third-party
client binaries — replicate the recipe on our own authentic APK.
