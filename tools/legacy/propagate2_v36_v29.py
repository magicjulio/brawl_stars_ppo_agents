#!/usr/bin/env python3
"""
propagate2_v36_v29.py — call-graph propagation v36(symbolized) -> v29(stripped).

Most Logic functions are ANCHORLESS (no rare 32-bit constants), so constant matching can't reach
them. This propagates identity through the call graph from a set of confirmed seed pairs:

  For a matched pair (f36,f29), take the ORDERED list of their call targets (bl/blx to known
  function entries). Align the two lists with difflib over a per-callee structural SIGNATURE
  (version-stable: has-push-prologue, token-size bucket, call-count, anchor-count). For each aligned
  callee pair, commit a match when it is unambiguous:
     * STRONG  : callee shares a rare constant anchor (near-unique id), OR
     * the aligned block length >= 2 (context lock), OR
     * the callee signature is locally unique on BOTH sides of this call list.
  Newly matched callees are enqueued; BFS continues. Anchor sharing also seeds cross-links.

Boundaries: v36 exact from labels; v29 = push{..lr} U bl-targets U post-terminator starts (same as
match3; avoids giant blobs). Validated on the executeOneTick subtree (tick/calcIsGameOver/calcXpBonuses).

Usage:
  python3 -I tools/propagate2_v36_v29.py --v36asm scratch/mapping/v36.asm --v29asm scratch/mapping/v29.asm \
      --targets 0x..,0x.. [--selftest] -o docs/offsets.v29.propagated.json
"""
import argparse, json, re, sys
from bisect import bisect_right
from collections import defaultdict, deque
from difflib import SequenceMatcher

LABEL = re.compile(r'^([0-9a-f]+)\s+<(.+)>:')
LINE  = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)(?:\s*@.*)?$')
BLT   = re.compile(r'^(?:bl|blx)\s+0x([0-9a-f]+)')
PUSH_LR = re.compile(r'^push\b.*\blr[\},]')
IMM   = re.compile(r'#(0x[0-9a-fA-F]+|\d+)')
MOVW  = re.compile(r'^movw\s+(r\d+|sp|lr),\s*#(0x[0-9a-f]+|\d+)')
MOVT  = re.compile(r'^movt\s+(r\d+|sp|lr),\s*#(0x[0-9a-f]+|\d+)')

# Confirmed/high seed pairs (v36 -> v29) from docs/offsets.v29.json
SEEDS = {
    0x8a26d8: 0x3198c4,  # executeOneTick CONFIRMED
    0x3274e4: 0x17b938,  # addGameObjects HIGH (constant match)
}
# Known answers for self-test (executeOneTick subtree, found earlier by position)
KNOWN = {
    0x1ee63c: 0x2bd008,  # tick
    0x8c9484: 0x1c8bdc,  # calculateIsGameOver
    0x22b910: 0x5dc818,  # calculateXpBonuses
}

def anchors_of(instrs):
    s = set(); movw = {}
    for mn, ops in instrs:
        line = (mn + ' ' + ops).strip()
        mw = MOVW.match(line)
        if mw:
            try: movw[mw.group(1)] = int(mw.group(2), 0)
            except ValueError: pass
            continue
        mt = MOVT.match(line)
        if mt:
            r = mt.group(1)
            if r in movw:
                try: s.add((((int(mt.group(2),0)&0xffff)<<16)|(movw[r]&0xffff)))
                except ValueError: pass
            continue
        for t in IMM.findall(ops):
            try: v = int(t,0)&0xffffffff
            except ValueError: continue
            if v >= 0x10000: s.add(v)
    return s

def parse_v36(path):
    funcs = {}  # addr -> dict(name, calls[list addr], size, push, anchors)
    a=None; nm=None; ins=[]; calls=[]; push=False
    def flush():
        if a is not None:
            funcs[a] = dict(name=nm, calls=calls, size=len(ins), push=push, anchors=anchors_of(ins))
    for ln in open(path, errors='replace'):
        m = LABEL.match(ln)
        if m:
            flush(); a=int(m.group(1),16); nm=m.group(2); ins=[]; calls=[]; push=False; continue
        if a is None: continue
        mi = LINE.match(ln)
        if not mi: continue
        mn=mi.group(2).lower()
        if mn.startswith('.') or mn=='word': continue
        ops=mi.group(3).strip().lower()
        ins.append((mn,ops))
        if len(ins)<=2 and PUSH_LR.match(mn+' '+ops): push=True
        b=BLT.match(mn+' '+ops)
        if b: calls.append(int(b.group(1),16))
    flush()
    return funcs

def parse_v29(path):
    rows=[]; bl=set()
    for ln in open(path, errors='replace'):
        mi=LINE.match(ln)
        if not mi: continue
        try: a=int(mi.group(1),16)
        except ValueError: continue
        mn=mi.group(2).lower()
        if mn.startswith('.') or mn=='word': continue
        ops=mi.group(3).strip().lower()
        is_push=bool(PUSH_LR.match(mn+' '+ops))
        b=BLT.match(mn+' '+ops); tgt=int(b.group(1),16) if b else None
        if tgt is not None: bl.add(tgt)
        is_term=(mn=='bx' and 'lr' in ops) or (mn=='pop' and 'pc' in ops)
        rows.append((a,mn,ops,is_push,tgt,is_term))
    lo,hi=rows[0][0],rows[-1][0]; addrset={r[0] for r in rows}
    entries=set(); prev=False
    for (a,mn,ops,is_push,tgt,is_term) in rows:
        if is_push: entries.add(a)
        if prev: entries.add(a)
        prev=is_term
    entries|={t for t in bl if lo<=t<=hi and t in addrset}
    entries=sorted(entries); estarts=entries
    def enc(a):
        i=bisect_right(estarts,a)-1
        return estarts[i] if i>=0 else None
    funcs={e:dict(calls=[], ins=[], push=False) for e in entries}
    for (a,mn,ops,is_push,tgt,is_term) in rows:
        e=enc(a)
        if e is None: continue
        f=funcs[e]; f['ins'].append((mn,ops))
        if is_push and a-e<=4 and len(f['ins'])<=2: f['push']=True
        if tgt is not None: f['calls'].append(tgt)
    for e in entries:
        f=funcs[e]; f['size']=len(f['ins']); f['anchors']=anchors_of(f['ins']); f['entryset']=None
    return entries, funcs

def sig(push, size, ncalls, nanch):
    return (bool(push), min(size//8,24), min(ncalls,16), min(nanch,6))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v36asm',required=True); ap.add_argument('--v29asm',required=True)
    ap.add_argument('--targets',default=''); ap.add_argument('--selftest',action='store_true')
    ap.add_argument('-o','--out',default='')
    a=ap.parse_args()
    sys.stderr.write('[p2] parse v36...\n'); v36=parse_v36(a.v36asm)
    sys.stderr.write('[p2] parse v29...\n'); v29e,v29=parse_v29(a.v29asm)
    v29set=set(v29e); v36set=set(v36)
    sys.stderr.write(f'[p2] v36={len(v36)} v29={len(v29e)}\n')

    # restrict call lists to entries/labels (real functions)
    for f in v36.values(): f['calls']=[c for c in f['calls'] if c in v36set]
    for e in v29e: v29[e]['calls']=[c for c in v29[e]['calls'] if c in v29set]

    # v29 anchor -> entries (for cross-link disambiguation)
    inv=defaultdict(list)
    for e in v29e:
        for k in v29[e]['anchors']: inv[k].append(e)

    def v36sig(c): f=v36[c]; return sig(f['push'], f['size'], len(f['calls']), len(f['anchors']))
    def v29sig(c): f=v29[c]; return sig(f['push'], f['size'], len(f['calls']), len(f['anchors']))

    match=dict(SEEDS); used={v for v in SEEDS.values()}; q=deque(SEEDS.keys())
    conf={k:'seed' for k in SEEDS}
    def commit(c36,c29,how):
        if c36 in match or c29 in used: return
        match[c36]=c29; used.add(c29); conf[c36]=how; q.append(c36)

    while q:
        a36=q.popleft(); a29=match.get(a36)
        if a29 is None or a36 not in v36 or a29 not in v29: continue
        L36=v36[a36]['calls']; L29=v29[a29]['calls']
        if not L36 or not L29: continue
        S36=[v36sig(c) for c in L36]; S29=[v29sig(c) for c in L29]
        sm=SequenceMatcher(a=S36,b=S29,autojunk=False)
        for i,j,n in sm.get_matching_blocks():
            if n==0: continue
            for k in range(n):
                c36=L36[i+k]; c29=L29[j+k]
                if c36 in match or c29 in used: continue
                # disambiguation
                sh = v36[c36]['anchors'] & v29[c29]['anchors']
                locuniq = (S36.count(S36[i+k])==1 and S29.count(S29[j+k])==1)
                if sh: commit(c36,c29,'anchor+cg')
                elif n>=2: commit(c36,c29,'cg-block')
                elif locuniq: commit(c36,c29,'cg-uniq')
        # anchor cross-links: callees of a36 with a unique shared anchor
        for c36 in L36:
            if c36 in match: continue
            anч=v36[c36]['anchors']
            for kk in anч:
                cand=[e for e in inv.get(kk,[]) if e not in used]
                if len(cand)==1 and len(inv.get(kk,[]))==1:
                    commit(c36,cand[0],'anchor'); break

    sys.stderr.write(f'[p2] total matched: {len(match)}\n')
    if a.selftest:
        ok=0
        for k,v in KNOWN.items():
            got=match.get(k)
            st='PASS' if got==v else 'FAIL'
            if got==v: ok+=1
            sys.stderr.write(f'[p2][selftest] {v36.get(k,{}).get("name","?")[:40]:40} want={hex(v)} got={hex(got) if got else None} {conf.get(k,"-")} {st}\n')
        sys.stderr.write(f'[p2][selftest] {ok}/{len(KNOWN)}\n')
    tgts=[int(x,0) for x in a.targets.split(',') if x.strip()]
    out={}
    for t in tgts:
        tt = t if t in v36 else (t & ~1)
        nm=v36.get(tt,{}).get('name',hex(t)); got=match.get(tt)
        out[nm]={'v36':hex(tt),'v29':(hex(got) if got else None),'how':conf.get(tt)}
        sys.stderr.write(f'[p2] {nm[:50]:50} -> {hex(got) if got else None} ({conf.get(tt)})\n')
    if a.out:
        json.dump(out,open(a.out,'w'),indent=2); sys.stderr.write(f'[p2] wrote {a.out}\n')

if __name__=='__main__':
    main()
