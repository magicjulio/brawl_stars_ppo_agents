#!/usr/bin/env python3
"""
extract_offsets.py — Phase 0b tooling for the headless-sim RL plan.

Given a (symbolized) Brawl Stars libg.so, this:
  1. parses the .symtab (via `readelf -sW` + `c++filt`) into name -> (addr, size),
  2. selects the Logic functions that matter (battle lifecycle, observation accessors,
     command/action encoding, data loading), grouped by purpose,
  3. for small leaf accessors (const getters / `return &field`), disassembles the function
     body with `llvm-objdump` and recovers the struct field offset from the first
     `this`-relative load/addr instruction (r0 is the `this` pointer in the ARM AAPCS).

Output: a JSON inventory ({group: [ {name, addr, size, thumb, offset?, hint?} ]}).

This is version-independent by construction: it is debugged against our v36.218 oracle
(which has a rich symtab) and is meant to be re-run on the pre-Arxan v29 runtime binary,
where the *names* are the same but the *addresses/offsets* differ.

Usage:
    python3 tools/extract_offsets.py <libg.so> -o offsets.json
Requires: readelf, c++filt, llvm-objdump on PATH.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import OrderedDict

# --- symbol groups: (group name, list of regexes matched against the DEMANGLED name) ---
# Kept deliberately tight so llvm-objdump is only invoked on a bounded set of symbols.
GROUPS = OrderedDict([
    ("battle_lifecycle", [
        r"\bLogicBattleModeServer::executeOneTick\(",
        r"\bLogicBattleModeServer::addPlayer\(",
        r"\bLogicBattleModeServer::spawnHero\(",
        r"\bLogicBattleModeServer::addGameObjects\(",
        r"\bLogicBattleModeServer::setBotDifficulty\(",
        r"\bLogicBattleModeServer::setRandomSeed\(",
        r"\bLogicBattleModeServer::forceGameOver\(",
        r"\bLogicBattleModeServer::getTeamScore\(",
        r"\bLogicBattleModeServer::getCurrentMVP\(",
        r"\bLogicBattleModeServer::handleClientInput\(",
        r"\bLogicBattleModeServer::tickUpdate\(",
        r"\bLogicBattleModeServer::generateTileMap\(",
        r"\bLogicBattleModeServer::setLocation\(",
    ]),
    ("command_action", [
        r"\bLogicCommandManager::createCommand\(",
        r"\bLogicCommandManager::addCommand\(",
        r"\bLogicCommandManager::encodeCommand\(",
        r"\bLogicCommandManager::decodeCommand\(",
        r"\bLogicCommandManager::saveCommandToJSON\(",
        r"\bLogicCommandManager::loadCommandFromJSON\(",
        r"\bLogicCommand::decode\(",
        r"\bLogicCommand::setExecuteTick\(",
    ]),
    ("data_load", [
        r"\bLogicDataTable::loadTable\(",
        r"\bLogicDataTable::setTable\(",
        r"\bLogicDataTables::LogicDataTables\(",
        r"\bLogicDataTables::load\w*\(",
        r"\bLogicDataTables::createReferences\(",
        r"\bLogicGlobals::LogicGlobals\(",
    ]),
    # observation accessors — the small leaf getters we recover offsets from
    ("obs_accessors", [
        r"\bLogicGameObjectServer::getPosition\(\) const$",
        r"\bLogicGameObjectServer::getX\(\) const$",
        r"\bLogicGameObjectServer::getY\(\) const$",
        r"\bLogicGameObjectServer::getTileX\(\) const$",
        r"\bLogicGameObjectServer::getTileY\(\) const$",
        r"\bLogicGameObjectServer::getGlobalID\(\) const$",
        r"\bLogicCharacterServer::getState\(\) const$",
        r"\bLogicCharacterServer::getBotSkill\(\)",
        r"\bLogicCharacterServer::getPowerLevel\(\) const$",
        r"\bLogicCharacterServer::isMoving\(\) const$",
        r"\bLogicCharacterServer::getCharacterData\(\) const$",
        r"\bLogicCharacterServer::getActiveSkillShield\(\) const$",
    ]),
    # functions where a field is *written* — offset recoverable from a str [r0,#imm]
    ("obs_writers", [
        r"\bLogicCharacterServer::setMaxHealth\(int\)$",
        r"\bLogicCharacterServer::setState\(",
    ]),
])

# Only disassemble bodies up to this many bytes (leaf accessors are tiny).
MAX_LEAF_BYTES = 96

# recover a `this`(r0)-relative field offset from a single instruction line
RE_LDR_R0 = re.compile(r"\bldr[bhs]*\s+r\d+,\s*\[r0(?:,\s*#(0x[0-9a-fA-F]+|\d+))?\]")
RE_STR_R0 = re.compile(r"\bstr[bh]*\s+r\d+,\s*\[r0(?:,\s*#(0x[0-9a-fA-F]+|\d+))?\]")
RE_ADD_R0 = re.compile(r"\badd\s+r\d+,\s*r0,\s*#(0x[0-9a-fA-F]+|\d+)")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_symbols(so):
    """Return list of dicts {addr, size, name(demangled)} for FUNC symbols."""
    raw = run(["readelf", "-sW", so])
    # demangle the whole blob once (c++filt leaves non-mangled text untouched)
    demangled = run(["c++filt"], ) if False else None
    p = subprocess.run(["c++filt"], input=raw, capture_output=True, text=True)
    lines = p.stdout.splitlines()
    syms = []
    for ln in lines:
        # Num:  Value   Size Type  Bind Vis   Ndx Name...
        parts = ln.split(None, 7)
        if len(parts) < 8:
            continue
        if parts[3] != "FUNC":
            continue
        try:
            addr = int(parts[1], 16)
            size = int(parts[2], 0)
        except ValueError:
            continue
        name = parts[7].strip()
        if not name or addr == 0:
            continue
        syms.append({"addr": addr, "size": size, "name": name})
    return syms


def disasm_offset(so, addr, size):
    """Disassemble a leaf body and return (offset:int|None, hint:str)."""
    lo = addr & ~1  # clear thumb bit
    hi = lo + max(size, 8)
    out = run(["llvm-objdump", "-d",
               f"--start-address={hex(lo)}", f"--stop-address={hex(hi)}", so])
    for ln in out.splitlines():
        if ":" not in ln:
            continue
        for rx in (RE_LDR_R0, RE_STR_R0, RE_ADD_R0):
            m = rx.search(ln)
            if m:
                val = m.group(1)
                off = int(val, 0) if val else 0
                return off, ln.strip()
    return None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so", help="path to libg.so")
    ap.add_argument("-o", "--out", default="offsets.json")
    args = ap.parse_args()

    syms = parse_symbols(args.so)
    by_name = {}
    for s in syms:
        by_name.setdefault(s["name"], s)  # first wins (symtab lists dup ctors)

    result = OrderedDict()
    result["_meta"] = {"so": args.so, "symbol_count": len(syms)}
    for group, patterns in GROUPS.items():
        regs = [re.compile(p) for p in patterns]
        hits = []
        seen = set()
        for s in syms:
            if s["name"] in seen:
                continue
            if any(r.search(s["name"]) for r in regs):
                seen.add(s["name"])
                entry = {"name": s["name"], "addr": hex(s["addr"]),
                         "size": s["size"], "thumb": bool(s["addr"] & 1)}
                if group in ("obs_accessors", "obs_writers") and 0 < s["size"] <= MAX_LEAF_BYTES:
                    off, hint = disasm_offset(args.so, s["addr"], s["size"])
                    if off is not None:
                        entry["offset"] = hex(off)
                        entry["hint"] = hint
                hits.append(entry)
        result[group] = hits
        print(f"[{group}] {len(hits)} symbols", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
