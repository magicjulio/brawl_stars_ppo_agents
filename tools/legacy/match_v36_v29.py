#!/usr/bin/env python3
"""
match_v36_v29.py — automate the v36(symbolized) -> v29(stripped) function mapping.

Method (the one proven by hand on executeOneTick):
  1. For each v36 target function (addr+size+name from offsets.v36.json), disassemble it
     and extract a FINGERPRINT: the 32-bit constants it materializes via movw+movt pairs
     (these are logic magics / float bit-patterns, very distinctive and version-stable).
  2. Pre-index the cached full v29 .text disassembly: every movw+movt constant -> the set of
     enclosing v29 function starts (a `push {..., lr}` is the function-start heuristic).
  3. A v29 function that carries the SAME rare constant(s) as the v36 target is the match;
     score by how many constants agree and how globally rare each is in v29.

Outputs docs/offsets.v29.auto.json with v29 addr, matched v36 name, evidence, confidence.
Leaf functions with no movw/movt constant are reported as UNMATCHED (resolve via call graph).

Usage:
  python3 tools/match_v36_v29.py --v36 apk/extracted/lib/armeabi-v7a/libg.so \
     --v29asm scratch/mapping/v29.asm --offsets docs/offsets.v36.json \
     [--groups battle_lifecycle,command_action] [--name-substr LogicDataTable] -o docs/offsets.v29.auto.json
"""
import argparse, json, re, subprocess, sys
from collections import defaultdict

LINE = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)(?:\s*@.*)?$')
MOVW = re.compile(r'^movw\s+(r\d+),\s*#(0x[0-9a-f]+|\d+)')
MOVT = re.compile(r'^movt\s+(r\d+),\s*#(0x[0-9a-f]+|\d+)')
BL   = re.compile(r'^(?:bl|blx)\s+0x([0-9a-f]+)')
PUSH_LR = re.compile(r'^push\b.*\blr\}')

def parse_lines(text):
    """Yield (addr:int, mnem:str, ops:str)."""
    for ln in text.splitlines():
        m = LINE.match(ln)
        if not m:
            continue
        try:
            addr = int(m.group(1), 16)
        except ValueError:
            continue
        yield addr, m.group(2).lower(), m.group(3).strip().lower()

def consts_from(seq):
    """Given list of (addr,mnem,ops), return set of 32-bit consts from movw+movt (same reg, within window)."""
    out = set()
    pend = {}   # reg -> (idx, lo)
    for i,(addr,mnem,ops) in enumerate(seq):
        mw = MOVW.match(mnem+' '+ops)
        mt = MOVT.match(mnem+' '+ops)
        if mw:
            pend[mw.group(1)] = (i, int(mw.group(2),0))
        elif mt:
            reg = mt.group(1)
            if reg in pend and i - pend[reg][0] <= 6:
                hi = int(mt.group(2),0)
                out.add(((hi<<16)|pend[reg][1]) & 0xffffffff)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v36', required=True)
    ap.add_argument('--v29asm', required=True)
    ap.add_argument('--offsets', required=True)
    ap.add_argument('--groups', default='')
    ap.add_argument('--name-substr', default='')
    ap.add_argument('-o','--out', default='docs/offsets.v29.auto.json')
    args = ap.parse_args()

    # --- index v29 ---
    sys.stderr.write('[match] loading v29 asm...\n')
    v29 = open(args.v29asm, errors='replace').read()
    func_starts = []                 # sorted list of v29 function-start addrs
    const2funcs = defaultdict(set)   # const -> {func_start}
    const_count = defaultdict(int)   # const -> global occurrences
    calls = defaultdict(set)         # func_start -> {target addr}
    cur = None
    pend = {}
    seqlen = 0
    for addr, mnem, ops in parse_lines(v29):
        full = mnem+' '+ops
        if PUSH_LR.match(full):
            cur = addr; func_starts.append(addr); pend = {}
        mw = MOVW.match(full); mt = MOVT.match(full)
        if mw:
            pend[mw.group(1)] = (addr, int(mw.group(2),0))
        elif mt and cur is not None:
            reg = mt.group(1)
            if reg in pend:
                c = ((int(mt.group(2),0)<<16)|pend[reg][1]) & 0xffffffff
                const2funcs[c].add(cur); const_count[c]+=1
        b = BL.match(full)
        if b and cur is not None:
            calls[cur].add(int(b.group(1),16))
    func_starts.sort()
    sys.stderr.write(f'[match] v29: {len(func_starts)} func-starts, {len(const2funcs)} distinct movw/movt consts\n')

    # --- targets from offsets.v36.json ---
    data = json.load(open(args.offsets))
    groups = [g for g in args.groups.split(',') if g] or [k for k in data if not k.startswith('_')]
    results = {}
    for g in groups:
        if g not in data: continue
        out_g = []
        for fn in data[g]:
            name = fn['name']
            if args.name_substr and args.name_substr not in name:
                continue
            addr = int(fn['addr'],16) & ~1
            size = max(int(fn.get('size',0)), 8)
            dis = subprocess.run(['llvm-objdump','-d','--no-show-raw-insn',
                    f'--start-address={hex(addr)}', f'--stop-address={hex(addr+size)}', args.v36],
                    capture_output=True, text=True).stdout
            seq = list(parse_lines(dis))
            c36 = consts_from(seq)
            entry = {'name':name, 'v36_addr':fn['addr'], 'v36_size':fn.get('size'),
                     'n_consts':len(c36)}
            if not c36:
                entry['v29_addr']=None; entry['confidence']='UNMATCHED_no_const'
                out_g.append(entry); continue
            # score candidate v29 funcs
            score = defaultdict(float); shared = defaultdict(list)
            for c in c36:
                for f in const2funcs.get(c,()):
                    w = 1.0/const_count[c]
                    score[f]+=w; shared[f].append(hex(c))
            if not score:
                entry['v29_addr']=None; entry['confidence']='UNMATCHED_no_v29_const'
                out_g.append(entry); continue
            best = max(score, key=lambda f:(score[f], len(shared[f])))
            nshare = len(shared[best])
            uniq = any(const_count[int(c,16)]==1 for c in shared[best])
            conf = 'HIGH' if (uniq or nshare>=2) else 'MEDIUM'
            entry.update({'v29_addr':hex(best),
                          'shared_consts':sorted(set(shared[best])),
                          'n_shared':nshare, 'score':round(score[best],3),
                          'confidence':conf})
            out_g.append(entry)
        results[g]=out_g
        hi = sum(1 for e in out_g if e.get('confidence')=='HIGH')
        md = sum(1 for e in out_g if e.get('confidence')=='MEDIUM')
        un = len(out_g)-hi-md
        sys.stderr.write(f'[match] {g}: {len(out_g)} targets -> HIGH={hi} MED={md} UNMATCHED={un}\n')
    json.dump(results, open(args.out,'w'), indent=2)
    sys.stderr.write(f'[match] wrote {args.out}\n')

if __name__=='__main__':
    main()
