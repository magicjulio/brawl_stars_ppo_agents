# 02 — Guard neutralization attempt & Arxan finding

Date: 2026-10-06. Status: **static patching hit the Arxan wall — pivot needed.**

## What we tried

Static-patched `lib/armeabi-v7a/libg.so` (patcher: `tools/patch_guards.py`, repack: `tools/repack.sh`),
overwriting 24 function entries with ARM return-stubs / a tail-call thunk, then repack + debug-sign + install
to the offline `bs_api30` AVD. Patch set:

- 12 guard spawners (lv1_detect_frida_library/server, lv1_frida_killer create_*_thread; root-detect inits; memory-guard ctor) → return
- `g_createGameInstance` → tail-call `GameMain::getInstance`
- `guard_callback`, `getAntihackFlags`, `isEmulator`, `isRooted`, `isTampered`, `isRepackaged`, `isSessionTampered`, `hasHackerTools`, `isSideLoaded` → 0
- `isSignatureValid` → 1

## Crash progression (each run moved the failure deeper)

1. **Stock APK:** SIGSEGV (MAPERR) in `lv1_detect_frida_library … parse_proc_maps` on a guard thread, ~2.6 s.
2. **After 12 guard stubs + gates:** past that; new SIGSEGV (ACCERR) on the **main thread**, executing from `[stack]` via `memmove`.
3. **After adding `g_createGameInstance`→`getInstance` thunk + `guard_callback`:** SIGSEGV (ACCERR) with **fault addr = 0x2897e4**, i.e. `GameMain::getInstance` itself.

## Root cause: Arxan on-demand code protection

Runtime maps from `tombstone_02`:

```
be711000-bf2fffff r--   0       bef000  libg.so   ← first LOAD segment (.text) mapped READ-ONLY, no execute
bf337000-bf77afff rw-   c25000  444000  libg.so   ← data
```

- The ELF marks the first LOAD segment `R E`, but at runtime Arxan `mprotect`s it to `r--`. **The real `.text` is never executable in place.**
- Executing code runs from `<anonymous:…>` / `[stack]` buffers — Arxan decrypts pages on demand (its `sigsegv_handler` routines) into scratch memory and runs them there.
- So the ACCERR at `0x2897e4` is my thunk doing `bx 0x2897e4` — branching to the **in-place, non-executable** copy of `getInstance`. The game never jumps there; Arxan runs a decrypted copy elsewhere.

### Why static patching fights this build

1. Executed code is a **decrypted copy**, not the on-disk bytes. Patches only matter if they survive into the copy.
2. Patching bytes inside an **encrypted region corrupts the plaintext** after decryption → crashes.
3. The frida-detection threads and the **on-demand page-decryption handler share the same `lv1_*` subsystem**. Stubbing the `create_*_thread` / memory-guard ctor likely also disabled the legitimate decrypt-on-fault handler → `.text` stays `r--` → execution faults with ACCERR and nothing catches it. Frida-detection and code-protection are entangled; you can't cleanly remove one by stubbing the whole subsystem.

### Emulator factor

The original **stock** crash was already an Arxan guard choking (parse_proc_maps MAPERR) under ARM-on-x86 translation
(ndk_translation). Arxan's SIGSEGV-driven decryption and `/proc/self/maps` scanning are fragile under binary
translation. Community pre-patched clients *do* run on emulators, so Arxan is defeatable there — but only by properly
unwrapping it (the peterr.dev author used undisclosed "private tools"), not by naive `.text` byte-patching.

## Recommendation / decision point

Static byte-patching of `.text` is the wrong tool against this Arxan build. Realistic paths for the RL goal
(reading game state from the running process), best first:

1. **Build on a community pre-patched v36 client** (ShadowwDev/BrawlStars-v36 Mega client, or royale-brawl-v29).
   Arxan is already handled; it boots offline. Our actual work — reading the observation space from memory at
   runtime — doesn't require *us* to defeat Arxan, only a running, instrumentable client. **Lowest risk.**
2. **Runtime instrumentation on a real ARM device / arm64 Android**, not the x86 emulator — hook the *decrypted,
   in-memory* code after Arxan processes it, rather than patching the file. Needs a Frida-gadget injection that
   survives `frida_killer`, which is itself non-trivial on a stock client.
3. **Properly unwrap Arxan statically** (dump decrypted pages at runtime, rebuild an executable `.so`). Highest
   effort; this is the "private tools" path.

Note: the symbolized `libg.so` (full DWARF, 42k symbols) remains fully valid for *analysis* — struct layouts for the
observation space can be read from it regardless of which runtime path we choose. Arxan only affects *executing a
patched client*, not reading symbols.

## Artifacts left in place

- `tools/patch_guards.py`, `tools/repack.sh` — the patcher/repack (kept for reference; approach superseded).
- `apk/patched/bs36218-patched.apk` — the 24-function-patched build (boots, crashes in Arxan decryption; not usable).
- `apk/patched/libg.so` — patched lib (sha256 `ff746cbf…f683f8`).
- Tombstones analyzed: `tomb0` (stock), `tomb1`, `tomb2` (patched) in scratchpad.
- AVD `bs_api30` (API 30 google_apis x86_64) left running, offline.
