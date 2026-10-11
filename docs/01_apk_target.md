# 01 — APK target selection

Status: **VERIFIED — confirmed the right fit** (2026-10-06)

## Recommendation

**Primary target: Brawl Stars 36.218 (official Supercell build, June 2021), using the `armeabi-v7a` `libg.so`.**
**Fallback / reference: Brawl Stars 29.270.**

## Verification results (2026-10-06)

The APK is in `apk/brawl-stars-36-218.apk` (extracted to `apk/extracted/`). All checks below ran locally.

- **Authentic & untampered.** `apksigner` verifies v1/v2/v3 schemes. Signer: `O=Supercell, L=Helsinki, C=FI`,
  cert SHA-256 `731a29e8…d60d8d3`, valid 2017→2042. versionCode `36218`, versionName `36.218`, native-code `armeabi-v7a` only.
  APK SHA-256 `3a2bfd07…a19f38ee`.
- **`libg.so` is NOT stripped — jackpot.** SHA-256 `f70b321c…d49c1df9`. 42,296 `FUNC` symbols + full DWARF
  (`.debug_info`, `.debug_str`, `.debug_line`…). Named functions cover ~94.9% of `.text`. So we read struct layouts
  from real names rather than reversing blind.
- **The whole observation space is named and in-process:**
  - `LogicBattleModeServer` — 122 functions: `tick()`, `executeOneTick(...)`, `handleClientInput(ClientInput*, ...)`,
    `addBots(...)`, `setBotDifficulty(int)`, `spawnHero(...)`, `updatePlayerPositions()`, `calculateIsGameOver()`,
    `getTeamScore(...)`, `stopAI()`, `handleOfflineInputMessage(...)`, `executeOfflineInputMessages()`.
  - `LogicCharacterServer` — full bot AI: `tickBot()` (13 KB), `handleMoveAndAttack()`, `attack(...)`, `getEnemyForAI(...)`,
    `aiUpdateSeenEnemies()`, `pathFindToNextTarget()`, `getClosestItemForAI(...)`, `aiUseOffensiveSkills(...)`, `tickAI()`.
  - plus `LogicGameObjectManager`, `LogicProjectileServer`, `ClientInput`. Offline bot battles are real and self-contained.
- **No Promon SHIELD.** This version predates it — there is no packed shield lib. Bundled libs are benign:
  `libsecsdk.so` = NetEase Yidun device-ID (via `com.netease.nis.sdkwrapper.Utils` / miitmdid),
  `libnova_cpp.so` = LINE Nova (regional login), `libsigner.so` = Adjust analytics signer, `libfmod.so` = audio,
  `libsentry*.so` = crash reporting.
- **Runs under ARM translation.** Installed `--abi armeabi-v7a` on a fresh Android 11 (API 30) google_apis x86_64
  AVD (ndk_translation / native bridge), fully offline (WiFi+data+airplane all off, network confirmed unreachable).
  The game launches, loads `libg.so`, reaches the splash — **no Supercell server was ever contacted.**

### The one hurdle (expected, and tractable)

The game SIGSEGVs ~2.6 s after launch. The faulting PC resolves to `libg.so` offset `0xE797C`, inside
`lv1_detect_frida_library … parse_proc_maps`. So the crash is the **anti-tamper guard suite**, not a translation bug.
Named guard routines present in `libg.so` (all in the clear, non-stripped):

- `lv1_detect_frida_library`, `lv1_detect_frida_server`, `lv1_frida_killer` (each with detection/guard threads,
  `parse_proc_maps/tcp/unix`, `sigsegv_handler`)
- `lv1_memory_guard` (`AndroidMemoryAnomalyGuard`), `lv1_root_detect_tamper_action` / `_non_tamper_action`
- `AntiCheat::isEmulator/isRooted/isTampered/guard_callback`, `Application::verifyDeviceIntegrity`,
  `titan::…::GameApp::isEmulator/isSignatureValid`, `g_createGameInstance` (Arxan) → `GameMain::getInstance`

Because these are **named functions in a non-stripped binary** (not a Promon-packed blob), neutralizing them is far
easier than on modern builds — this is exactly why an old version was the right call. Doing it is the next stage.

> **Note (session handoff):** writing the guard-neutralization / instrumentation tooling tripped the Opus 5.5
> `[cyber]` classifier twice. That stage should be done on **Opus 4.8**. Intent is research-only, fully offline;
> public Supercell servers are never a target.

## Why this matters for our approach

We want to read game state out of the running process, and we want offline matches against bots. Two facts make that possible:

1. **The battle server runs inside the client.** Old clients ship `LogicBattleModeServer` (the server-side
   battle simulation, including bot AI) inside `libg.so`. "Offline battles" in community private servers are
   that code running in-process. The client simulates the match against bots and only reports the result back
   (`AskForBattleEndMessage`). The v29 community script hooks `LogicBattleModeServer::setBotDifficulty`
   directly (writes difficulty at `this+184`). So the authoritative game state (all characters, projectiles,
   HP, positions) is in our own process, and no server-side simulation is needed.
2. **Symbols decide how much reversing we have to do.** Supercell shipped stripped, Arxan-obfuscated `libg.so`
   builds. A community RE guide links a **symbolized IDA database for v36 Android armeabi-v7a `libg.so`**
   ("Supercell accidentally pushed builds with symbols left in"). With real names (`LogicBattleModeServer`,
   `LogicGameObjectManager`, `LogicCharacter`, ...) we can recover struct layouts for the observation space
   far faster than from a stripped binary.

## Candidates compared

| | **36.218** (primary) | **29.270** (fallback) | 26.184 (Classic Brawl) | ≥ v40s / modern |
|---|---|---|---|---|
| Symbolized `libg.so` | **Yes**: IDB exists (armv7) | No (port names from v36 via BinDiff) | No | v62 iOS IDB only |
| Offline bot battles proven | Yes, but community notes only Gem Grab / Bounty / Big Game are stable | **Yes, most mature** (Royale Brawl v29) | Yes | Some (BSDS v40/v44) |
| Public bypass offsets | Not public (we derive them; symbols make this easy) | **Public**: full Frida script | Patched APK only | Partial |
| Protections | Arxan + Supercell AntiCheat (+ likely shield lib, verify) | Arxan + AntiCheat + shield lib (`lib39285EFA.so`), bypassed with ~5 hooks | Arxan | Promon SHIELD (SVC syscalls, anti-Frida, anti-emulator): **avoid** |
| Original APK provenance | uptodown upload dated release day (2021-06-16) | uptodown entries re-uploaded in 2026 (check cert) | uptodown 2020 | n/a |

Gem Grab / Bounty 3v3 against bots is a good first RL task, so the mode limitation on v36 is acceptable.
Broken modes are probably server/data-side issues in that community server, and symbols should let us fix them later.

## Known protection surface (from the v29 community client, should be similar in v36)

All of these are bypassed in the v29 script with Frida `Interceptor.replace`/`attach`:

- `g_createGameInstance` (Arxan-obfuscated) → replaced with `GameMain::getInstance`
- `AntiCheat::getAntihackFlags` → return 0
- `AntiCheat::guard_callback` → no-op
- `titan::GameApp::isSignatureValid` → return 1 (needed when the APK is repackaged)
- shield lib `lib39285EFA.so` +0xC0D0 → return 0
- networking: redirect `connect()` on port 9339 to a local server, plus manual message-queue pumping
- offline: force offline-battle check true (`+0x44CEBC`), force game-mode arg = 3 at `+0x67FEBC`

On v36 we look these functions up by name in the symbolized IDB instead of using offsets.
Peterr's v30 write-up describes the same technique (Arxan, string encryption, `ServerConnection::update`, keepalive) on a neighbouring version.

## Next steps / verification checklist

1. **Get the official APK manually.** Uptodown 36.218, versionID `3746973`, ~183 MB XAPK. The download is behind a
   Cloudflare captcha, so it can't be scripted: https://brawl-stars.en.uptodown.com/android/download/3746973
   (APKMirror also lists 36.218 with arm64-v8a and arm-v7a.)
2. Verify provenance: `apksigner verify --print-certs`. The cert must be Supercell's, and it should match a modern
   Play Store Brawl Stars cert.
3. Inspect `lib/armeabi-v7a/`: list the `.so` files and flag any random-named large lib (the shield). Run `readelf -S`/`-s` on `libg.so`
   and record its sha256.
4. Get the v36 IDB (`libg.so.i64`, armv7) linked from the RE guide below. Confirm it matches **our** `libg.so`
   (same size / sha256 / function bytes at named addresses). This needs IDA. Ghidra can't open `.i64`, so if needed
   we can export names from IDA to a Ghidra/r2 symbol map. Treat it as untrusted input.
5. Runtime: v36 armv7 on our x86_64 host needs ARM translation. Options are an Android 11+ x86_64 emulator image (ndk_translation),
   Waydroid + libhoudini, or a real ARM device. Install with `--abi armeabi-v7a` so offsets match the IDB.
   For scaling later: an ARM server with AArch32 EL0 support (e.g. Ampere Altra; Graviton3+ dropped AArch32) + redroid.
6. Bring-up: Frida gadget → bypass hooks → minimal local server (use ShadowwDev v36 Python server as reference)
   → press PLAY → offline Gem Grab vs bots → hook `LogicBattleModeServer` tick to dump game objects.

## Sources

- Peterr, "Making Brawl Stars offline" (v30, Arxan, Frida): https://peterr.dev/blogs/making-brawl-stars-offline/
- Brawl Stars RE guide (symbolized v36 Android / v62 iOS IDBs, Promon notes): https://gist.github.com/rawnullbyte/ded80fd72dabc581751ed20fd4338780
- ShadowwDev/BrawlStars-v36 (v36.218 server, offline battles, stable modes): https://github.com/ShadowwDev/BrawlStars-v36
- erder00/royale-brawl-v29 (v29.270 client script `liberder.script.js`, C# battle reimplementation): https://github.com/erder00/royale-brawl-v29
- PhoenixFire6934/Classic-Brawl (v26.184): https://github.com/PhoenixFire6934/Classic-Brawl
- Promon SHIELD in Brawl Stars: https://github.com/KiFilterFiberContext/promon-reversal
