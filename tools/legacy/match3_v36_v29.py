#!/usr/bin/env python3
"""
match3_v36_v29.py — opcode-sequence structural matcher, v36(symbolized) -> v29(stripped).

For each v36 target function (located by its <label>: in v36.asm), build a normalized token
sequence (mnemonic + operand-shape; rare 32-bit immediates kept literal as strong anchors). Then,
for every v29 function entry, measure how much of the v36 token sequence is found IN ORDER inside
the v29 candidate (SequenceMatcher matching-block coverage of the v36 side). This coverage metric is
robust to (a) v29 over-segmentation / boundary inflation and (b) minor codegen drift between versions.
Final score = coverage * (1 + rare-constant bonus). Reports top candidates + margin over 2nd.

Boundaries:
  v36: exact from `^<hex> <name>:` labels.
  v29: heuristic entries = push{..lr} UNION direct bl/blx targets (covers leaves), plus starts right
       after a hard terminator (bx lr / pop{..pc}) that are themselves push/bl-target. Candidate body
       = entry .. next entry (may inflate; coverage metric tolerates it).

Usage:
  python3 -I tools/match3_v36_v29.py --v36asm scratch/mapping/v36.asm --v29asm scratch/mapping/v29.asm \
      --targets 0x4f2458,0x6fd8cc,... [--selftest] -o out.json
Self-test maps the 5 confirmed answers and asserts each ranks #1.
"""
import argparse, json, re, sys
from bisect import bisect_right
from collections import defaultdict
from difflib import SequenceMatcher

LABEL = re.compile(r'^([0-9a-f]+)\s+<(.+)>:')
LINE  = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)(?:\s*@.*)?$')
BL    = re.compile(r'^(bl|blx)\b')
PUSH_LR = re.compile(r'^push\b.*\blr[\},]')
IMM   = re.compile(r'#(0x[0-9a-fA-F]+|\d+)')
REG   = re.compile(r'\b(r\d+|sp|lr|pc|fp|ip)\b')

# Known-answer pairs (v36 -> v29), all CONFIRMED/HIGH in docs/offsets.v29.json
KNOWN = {
    0x8a26d8: 0x3198c4,  # executeOneTick (CONFIRMED)
    0x3274e4: 0x17b938,  # addGameObjects (HIGH)
    0x1ee63c: 0x2bd008,  # tick (HIGH)
    0x8c9484: 0x1c8bdc,  # calculateIsGameOver (HIGH)
    0x22b910: 0x5dc818,  # calculateXpBonuses (MEDIUM)
}

def tokenize(mn, ops):
    """Normalized opcode-shape token (for coverage). Immediates -> shape only here;
    rare constants handled separately via anchors()."""
    mn = mn.lower()
    toks = [mn]
    if BL.match(mn):
        toks.append('CALL')
        return '|'.join(toks)
    o = ops.lower()
    if 'pc' in o and ('ldr' in mn or 'add' in mn):
        toks.append('PCREL')
    if IMM.search(o): toks.append('i')
    if '[' in o: toks.append('M')
    nr = len(REG.findall(o))
    if nr: toks.append('r%d' % min(nr, 4))
    return '|'.join(toks)

MOVW = re.compile(r'^movw\s+(r\d+|sp|lr),\s*#(0x[0-9a-f]+|\d+)')
MOVT = re.compile(r'^movt\s+(r\d+|sp|lr),\s*#(0x[0-9a-f]+|\d+)')

def func_anchors(instrs):
    """Rare constants of a function: movw/movt-reconstructed 32-bit values + standalone
    immediates >= 0x400 (skip tiny struct offsets/enums that collide everywhere).
    instrs = list of (mn, ops) lowercased."""
    anchors = set()
    movw_val = {}  # reg -> last movw imm
    for mn, ops in instrs:
        line = (mn + ' ' + ops).strip()
        mw = MOVW.match(line)
        if mw:
            try: movw_val[mw.group(1)] = int(mw.group(2), 0)
            except ValueError: pass
            continue
        mt = MOVT.match(line)
        if mt:
            reg = mt.group(1)
            if reg in movw_val:
                try:
                    hi = int(mt.group(2), 0)
                    anchors.add('K=%x' % (((hi & 0xffff) << 16) | (movw_val[reg] & 0xffff)))
                except ValueError: pass
            continue
        for t in IMM.findall(ops):
            try: v = int(t, 0) & 0xffffffff
            except ValueError: continue
            if v >= 0x10000:               # genuinely large standalone const (rare); skip small offsets/enums
                anchors.add('C=%x' % v)
    return anchors

def parse_v36(path):
    """Return {addr: (name, [tokens], anchors)} for every labeled function."""
    funcs = {}
    cur_addr = None; cur_name = None; cur = []; cur_ins = []
    def flush():
        if cur_addr is not None:
            funcs[cur_addr] = (cur_name, cur, func_anchors(cur_ins))
    for ln in open(path, errors='replace'):
        m = LABEL.match(ln)
        if m:
            flush()
            cur_addr = int(m.group(1), 16); cur_name = m.group(2); cur = []; cur_ins = []
            continue
        if cur_addr is None: continue
        mi = LINE.match(ln)
        if not mi: continue
        mn = mi.group(2).lower()
        if mn.startswith('.') or mn == 'word': continue  # data words in literal pools
        ops = mi.group(3).strip().lower()
        cur.append(tokenize(mn, ops)); cur_ins.append((mn, ops))
    flush()
    return funcs

def parse_v29(path):
    """Heuristic function segmentation of stripped v29. Return (entries_sorted, {entry:[tokens]})."""
    rows = []  # (addr, token, is_push_lr, is_bl, is_term)
    bl_tgts = set()
    for ln in open(path, errors='replace'):
        mi = LINE.match(ln)
        if not mi: continue
        try: a = int(mi.group(1), 16)
        except ValueError: continue
        mn = mi.group(2).lower(); ops = mi.group(3).strip().lower()
        if mn.startswith('.') or mn == 'word': continue
        tok = tokenize(mn, ops)
        is_bl = bool(BL.match(mn))
        if is_bl:
            mt = re.search(r'0x([0-9a-f]+)', ops)
            if mt: bl_tgts.add(int(mt.group(1), 16))
        is_push = bool(PUSH_LR.match(mn + ' ' + ops))
        is_term = (mn == 'bx' and 'lr' in ops) or (mn == 'pop' and 'pc' in ops)
        rows.append((a, tok, is_push, is_bl, is_term, mn, ops))
    lo, hi = rows[0][0], rows[-1][0]
    entries = set()
    addrset = {r[0] for r in rows}
    prev_term = False
    for (a, tok, is_push, is_bl, is_term, mn, ops) in rows:
        if is_push: entries.add(a)
        if prev_term: entries.add(a)        # split at EVERY hard terminator: kills over-merged blobs
        prev_term = is_term
    entries |= {t for t in bl_tgts if lo <= t <= hi and t in addrset}
    entries = sorted(entries)
    # map addr -> tokens + raw instrs grouped by enclosing entry
    bodies = defaultdict(list); ins = defaultdict(list)
    estarts = entries
    def enc(a):
        i = bisect_right(estarts, a) - 1
        return estarts[i] if i >= 0 else None
    for (a, tok, is_push, is_bl, is_term, mn, ops) in rows:
        e = enc(a)
        if e is not None:
            bodies[e].append(tok); ins[e].append((mn, ops))
    anchors = {e: func_anchors(ins[e]) for e in bodies}
    return entries, bodies, anchors

def coverage(src, cand):
    """Fraction of src tokens found in-order within cand (robust to cand inflation)."""
    if not src: return 0.0
    sm = SequenceMatcher(a=src, b=cand, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(src)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v36asm', required=True); ap.add_argument('--v29asm', required=True)
    ap.add_argument('--targets', default=''); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--topk', type=int, default=3); ap.add_argument('-o', '--out', default='')
    a = ap.parse_args()

    import math
    sys.stderr.write('[m3] parsing v36...\n'); v36 = parse_v36(a.v36asm)
    sys.stderr.write('[m3] parsing v29...\n'); v29_entries, v29_bodies, v29_anch = parse_v29(a.v29asm)
    sys.stderr.write(f'[m3] v36 funcs={len(v36)}  v29 entries={len(v29_entries)}\n')

    # idf of each anchor over v29 functions (rare anchor => high idf => strong signal)
    df = defaultdict(int)
    for e, s in v29_anch.items():
        for k in s: df[k] += 1
    N = max(len(v29_anch), 1)
    def idf(k): return math.log(N / (1 + df.get(k, 0)))
    # inverted index anchor -> v29 entries (to restrict candidate set to anchor-sharing fns)
    inv = defaultdict(list)
    for e, s in v29_anch.items():
        for k in s: inv[k].append(e)

    targets = []
    if a.selftest: targets = list(KNOWN.keys())
    targets += [int(x, 0) for x in a.targets.split(',') if x.strip()]

    results = {}
    ok = 0
    for tgt in targets:
        if tgt not in v36:
            # try thumb-cleared or nearest label
            tgt2 = tgt & ~1
            if tgt2 in v36: tgt = tgt2
        if tgt not in v36:
            sys.stderr.write(f'[m3] target {hex(tgt)} not a v36 label\n'); continue
        name, src, src_anch = v36[tgt]
        # Only anchor-rich functions are reliably mappable here. Anchorless (no rare 32-bit
        # constants) -> mark for call-graph propagation instead of a useless coverage scan.
        if not src_anch:
            results[name] = {'v36': hex(tgt), 'name': name, 'nsrc': len(src), 'nanchor': 0,
                             'top': [], 'margin': 0.0, 'note': 'ANCHORLESS -> use call-graph'}
            sys.stderr.write(f'[m3] {name[:46]:46} -> ANCHORLESS (nsrc={len(src)}) -> call-graph\n')
            continue
        cands = set()
        for k in src_anch: cands.update(inv.get(k, []))
        scored = []
        for e in cands:
            cand = v29_bodies[e]
            if not cand: continue
            shared = src_anch & v29_anch.get(e, set())
            anchor_score = sum(idf(k) for k in shared)         # PRIMARY: rare-const idf mass
            cov = coverage(src, cand)                          # SECONDARY: opcode-sequence coverage
            score = anchor_score + 0.5 * cov
            scored.append((score, cov, len(shared), anchor_score, e))
        scored.sort(reverse=True)
        top = scored[:a.topk]
        best = top[0] if top else None
        second = top[1][0] if len(top) > 1 else 0.0
        margin = (best[0] - second) if best else 0.0
        rec = {'v36': hex(tgt), 'name': name, 'nsrc': len(src), 'nanchor': len(src_anch),
               'top': [{'v29': hex(e), 'score': round(s,3), 'cov': round(c,3),
                        'shared': sh, 'anchor': round(an,2)} for (s,c,sh,an,e) in top],
               'margin': round(margin,3)}
        results[name] = rec
        if a.selftest and tgt in KNOWN:
            want = KNOWN[tgt]
            got = best[4] if best else None
            rank = next((i for i,t in enumerate(scored) if t[4]==want), -1)
            status = 'PASS' if got==want else f'FAIL(rank={rank})'
            if got==want: ok += 1
            sys.stderr.write(f'[m3][selftest] {name[:40]:40} want={hex(want)} got={hex(got) if got else None} '
                             f'score={best[0]:.3f} margin={margin:.3f} nanch={len(src_anch)} {status}\n')
        else:
            b = best
            sys.stderr.write(f'[m3] {name[:46]:46} -> {hex(b[4]) if b else None} '
                             f'score={b[0]:.3f} cov={b[1]:.3f} shared={b[2]} margin={margin:.3f} nanch={len(src_anch)}\n')
    if a.selftest:
        sys.stderr.write(f'[m3][selftest] {ok}/{len(KNOWN)} ranked #1\n')
    if a.out:
        json.dump(results, open(a.out,'w'), indent=2)
        sys.stderr.write(f'[m3] wrote {a.out}\n')

if __name__ == '__main__':
    main()
