#!/usr/bin/env python3
"""
propagate_v36_v29.py — call-graph propagation to map v36(symbolized)->v29(stripped).

Builds both call graphs from cached full .text disassembly, seeds with high-confidence
constant matches, then iteratively: for each matched caller pair whose ordered call-target
lists have equal length, maps callees positionally and validates by structure (prologue
register-list + size + call-count). v36 callees are labeled via the symtab; any target
function reached is emitted with its v29 address.

Function entries = every bl/blx target (covers leaf fns that lack a push prologue) ∪ push{..lr}.

Usage:
  python3 tools/propagate_v36_v29.py --v36asm scratch/mapping/v36.asm \
     --v29asm scratch/mapping/v29.asm --syms scratch/mapping/v36_syms.tsv \
     --offsets docs/offsets.v36.json --seeds docs/offsets.v29.auto.json \
     -o docs/offsets.v29.propagated.json
"""
import argparse, json, re, sys
from bisect import bisect_right
from collections import defaultdict, deque
from difflib import SequenceMatcher

LINE = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)(?:\s*@.*)?$')
BL   = re.compile(r'^(?:bl|blx)\s+0x([0-9a-f]+)')
PUSH_LR = re.compile(r'^push\b.*\blr[\},]')

def load(path):
    instrs=[]; bl_tgts=set(); push=set(); after_ret=set()
    prev_term=False
    for ln in open(path, errors='replace'):
        m=LINE.match(ln)
        if not m: continue
        try: a=int(m.group(1),16)
        except ValueError: continue
        mn=m.group(2).lower(); ops=m.group(3).strip().lower()
        if prev_term: after_ret.add(a)           # start of a new function after a hard return
        instrs.append((a,mn,ops))
        full=mn+' '+ops
        b=BL.match(full)
        if b: bl_tgts.add(int(b.group(1),16))
        if PUSH_LR.match(full): push.add(a)
        # hard terminators: bx lr  |  pop {...,pc}
        prev_term = ((mn=='bx' and 'lr' in ops) or (mn=='pop' and 'pc' in ops))
    lo=instrs[0][0]; hi=instrs[-1][0]
    # entries = call targets ∪ prologues ∪ post-return starts, but only those that begin a
    # real function (push prologue OR are a direct bl target) to avoid over-segmenting on
    # intra-function unconditional returns.
    entries=sorted(({t for t in bl_tgts if lo<=t<=hi} | push) | (after_ret & (push | bl_tgts)))
    return instrs, entries, bl_tgts

def build(instrs, entries):
    """Return calls[entry]=ordered list of call-target addrs, and feats[entry]=(size,push_regs,ncalls)."""
    estarts=entries
    calls=defaultdict(list); push_regs={}
    # map addr->enclosing entry via bisect
    def enc(a):
        i=bisect_right(estarts,a)-1
        return estarts[i] if i>=0 else None
    # record prologue reg-list: first push{..lr} at/after entry (within a few insns)
    for idx,(a,mn,ops) in enumerate(instrs):
        if mn=='push' and 'lr' in ops:
            e=enc(a)
            if e is not None and e not in push_regs and a-e<=8:
                push_regs[e]=ops
        b=BL.match(mn+' '+ops)
        if b:
            e=enc(a)
            if e is not None: calls[e].append(int(b.group(1),16))
    feats={}
    for i,e in enumerate(estarts):
        nxt=estarts[i+1] if i+1<len(estarts) else e+64
        feats[e]=(nxt-e, push_regs.get(e), len(calls.get(e,[])))
    return calls, feats

def compatible(f36, f29):
    s36,p36,n36=f36; s29,p29,n29=f29
    # both leaf (no push) or both have push with same reg-list length-ish
    if (p36 is None)!=(p29 is None): return False
    if n36!=n29: return False               # require equal direct-call count
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v36asm',required=True); ap.add_argument('--v29asm',required=True)
    ap.add_argument('--syms',required=True); ap.add_argument('--offsets',required=True)
    ap.add_argument('--seeds',required=True); ap.add_argument('-o','--out',required=True)
    a=ap.parse_args()

    sys.stderr.write('[prop] loading v36...\n'); i36,e36,_=load(a.v36asm)
    sys.stderr.write('[prop] loading v29...\n'); i29,e29,_=load(a.v29asm)
    sys.stderr.write(f'[prop] entries v36={len(e36)} v29={len(e29)}\n')
    c36,f36=build(i36,e36); c29,f29=build(i29,e29)
    e36set=set(e36); e29set=set(e29)

    # symtab addr->name (thumb bit cleared)
    name={}
    for ln in open(a.syms, errors='replace'):
        t=ln.rstrip('\n').split('\t',1)
        if len(t)!=2: continue
        try: ad=int(t[0],16)&~1
        except ValueError: continue
        if ad: name.setdefault(ad,t[1])

    # seeds: confirmed + constant-HIGH
    match={}  # v36_addr -> v29_addr
    seeds=[(0x8a26d8,0x3198c4),(0x3274e4,0x17b938)]  # executeOneTick, addGameObjects (confirmed)
    try:
        sd=json.load(open(a.seeds))
        for g in sd:
            for ent in sd[g]:
                if ent.get('confidence')=='HIGH' and ent.get('v29_addr'):
                    seeds.append((int(ent['v36_addr'],16)&~1, int(ent['v29_addr'],16)))
    except Exception as ex:
        sys.stderr.write(f'[prop] seeds load warn: {ex}\n')
    q=deque()
    for v36a,v29a in seeds:
        if v36a in e36set and v29a in e29set and v36a not in match:
            match[v36a]=v29a; q.append(v36a)
    sys.stderr.write(f'[prop] seeds={len(match)}\n')

    def sig(feats_map, e):
        s,p,n = feats_map.get(e,(0,None,0))
        return (p is not None, min(n,20), min(s//16,64))

    used29=set(match.values())
    # BFS propagate by difflib-aligned call-site matching (tolerates insert/delete)
    accepted=0
    while q:
        a36=q.popleft(); a29=match[a36]
        L36=[t&~1 for t in c36.get(a36,[])]; L29=[t&~1 for t in c29.get(a29,[])]
        if not L36 or not L29: continue
        S36=[sig(f36,t) for t in L36]; S29=[sig(f29,t) for t in L29]
        sm=SequenceMatcher(a=S36,b=S29,autojunk=False)
        for i,j,n in sm.get_matching_blocks():
            if n==0: continue
            for k in range(n):
                t36=L36[i+k]; t29=L29[j+k]
                if t36 not in e36set or t29 not in e29set: continue
                if t36 in match or t29 in used29: continue
                # commit if block>=2 OR this signature is locally unique on both sides
                locuniq = (S36.count(S36[i+k])==1 and S29.count(S29[j+k])==1)
                if n>=2 or locuniq:
                    match[t36]=t29; used29.add(t29); q.append(t36); accepted+=1
    sys.stderr.write(f'[prop] propagated matches: {accepted} (total {len(match)})\n')

    # emit targets from offsets.v36.json
    data=json.load(open(a.offsets)); out={}
    for g in data:
        if g.startswith('_'): continue
        rows=[]
        for fn in data[g]:
            v36a=int(fn['addr'],16)&~1
            v29a=match.get(v36a)
            rows.append({'name':fn['name'],'v36_addr':fn['addr'],
                         'v29_addr':(hex(v29a) if v29a else None),
                         'via':('seed/propagation' if v29a else None)})
        hit=sum(1 for r in rows if r['v29_addr'])
        out[g]=rows
        sys.stderr.write(f'[prop] {g}: {hit}/{len(rows)} mapped\n')
    json.dump(out, open(a.out,'w'), indent=2)
    sys.stderr.write(f'[prop] wrote {a.out}\n')

if __name__=='__main__':
    main()
