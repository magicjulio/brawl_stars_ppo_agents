#!/usr/bin/env python3
"""
match2_v36_v29.py — structural function matcher v36(symbolized)->v29(stripped), using the
tf-idf similarity of each function's IMMEDIATE-VALUE multiset (all #imm operands: struct
field offsets, enum/threshold constants, magic numbers), which is distinctive and largely
version-stable for these struct-heavy Logic functions. Generalizes the movw/movt-only method.

v29 function entries = every bl/blx target ∪ push{..,lr} (covers leaves). An inverted index
(imm -> v29 entries) makes scoring fast; rare immediates (high idf) dominate the score.
Confidence from absolute shared-idf mass and the margin over the 2nd-best candidate, with a
call-count / prologue compatibility check as a guard.

Usage:
  python3 tools/match2_v36_v29.py --v36 <libg36.so> --v29asm scratch/mapping/v29.asm \
     --offsets docs/offsets.v36.json [--groups ...] [--name-substr ...] -o out.json
"""
import argparse, json, re, subprocess, sys, math
from bisect import bisect_right
from collections import defaultdict, Counter

LINE = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*?)(?:\s*@.*)?$')
IMM  = re.compile(r'#(0x[0-9a-fA-F]+|\d+)')
BL   = re.compile(r'^(?:bl|blx)\s+0x([0-9a-f]+)')
PUSH_LR = re.compile(r'^push\b.*\blr[\},]')

def imms(ops):
    out=[]
    for t in IMM.findall(ops):
        try: out.append(int(t,0)&0xffffffff)
        except ValueError: pass
    return out

def parse(path_or_text, is_text=False):
    src = path_or_text.splitlines() if is_text else open(path_or_text, errors='replace')
    rows=[]
    for ln in src:
        m=LINE.match(ln)
        if not m: continue
        try: a=int(m.group(1),16)
        except ValueError: continue
        rows.append((a, m.group(2).lower(), m.group(3).strip().lower()))
    return rows

def index_v29(rows):
    bl=set(); push=set()
    for a,mn,ops in rows:
        full=mn+' '+ops
        b=BL.match(full)
        if b: bl.add(int(b.group(1),16))
        if PUSH_LR.match(full): push.add(a)
    lo,hi=rows[0][0],rows[-1][0]
    entries=sorted({t for t in bl if lo<=t<=hi}|push)
    def enc(a):
        i=bisect_right(entries,a)-1
        return entries[i] if i>=0 else None
    imm_ms=defaultdict(Counter); ncalls=Counter(); has_push={}
    for a,mn,ops in rows:
        e=enc(a)
        if e is None: continue
        full=mn+' '+ops
        for v in imms(ops): imm_ms[e][v]+=1
        if BL.match(full): ncalls[e]+=1
        if mn=='push' and 'lr' in ops and a-e<=8: has_push[e]=True
    # df over entries
    df=Counter()
    for e,ms in imm_ms.items():
        for v in ms: df[v]+=1
    N=len(entries)
    idf={v:math.log(N/df[v]) for v in df}
    inv=defaultdict(list)
    for e,ms in imm_ms.items():
        for v in ms: inv[v].append(e)
    return entries, imm_ms, ncalls, has_push, idf, inv, N

def sig_from_v36(so, addr, size):
    size=max(size,8)
    dis=subprocess.run(['llvm-objdump','-d','--no-show-raw-insn',
        f'--start-address={hex(addr)}',f'--stop-address={hex(addr+size)}', so],
        capture_output=True,text=True).stdout
    rows=parse(dis, is_text=True)
    ms=Counter(); nc=0; hp=False
    for a,mn,ops in rows:
        for v in imms(ops): ms[v]+=1
        full=mn+' '+ops
        if BL.match(full): nc+=1
        if mn=='push' and 'lr' in ops: hp=True
    return ms,nc,hp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v36',required=True); ap.add_argument('--v29asm',required=True)
    ap.add_argument('--offsets',required=True); ap.add_argument('--groups',default='')
    ap.add_argument('--name-substr',default=''); ap.add_argument('-o','--out',required=True)
    a=ap.parse_args()
    sys.stderr.write('[m2] indexing v29...\n')
    entries,imm_ms,ncalls,has_push,idf,inv,N=index_v29(parse(a.v29asm))
    sys.stderr.write(f'[m2] v29 entries={N}, distinct imms={len(idf)}\n')

    data=json.load(open(a.offsets))
    groups=[g for g in a.groups.split(',') if g] or [k for k in data if not k.startswith('_')]
    out={}
    for g in groups:
        if g not in data: continue
        rows=[]
        for fn in data[g]:
            if a.name_substr and a.name_substr not in fn['name']: continue
            v36a=int(fn['addr'],16)&~1
            ms36,nc36,hp36=sig_from_v36(a.v36, v36a, int(fn.get('size',8)))
            # candidate scores via inverted index, rare imms only (idf>0.5)
            score=defaultdict(float)
            for v,tf in ms36.items():
                if v not in inv or idf[v]<0.5: continue
                w=idf[v]
                for e in inv[v]:
                    score[e]+=min(tf, imm_ms[e][v])*w
            # guard: prefer call-count match + same push-ness
            ranked=sorted(score.items(), key=lambda kv:kv[1], reverse=True)
            best=None; conf='UNMATCHED'; margin=0; bscore=0
            for e,sc in ranked[:8]:
                if (ncalls[e]==nc36) and (has_push.get(e,False)==hp36):
                    best=e; bscore=sc; break
            if best is None and ranked:
                best,bscore=ranked[0]
            if ranked:
                second=ranked[1][1] if len(ranked)>1 else 0.0
                margin=bscore-second
                if bscore>=6 and (margin>=0.4*bscore or bscore>=20): conf='HIGH'
                elif bscore>=3: conf='MEDIUM'
            rows.append({'name':fn['name'],'v36_addr':fn['addr'],
                'v29_addr':(hex(best) if best else None),
                'score':round(bscore,2),'margin':round(margin,2),
                'nc36':nc36,'nc29':(ncalls[best] if best else None),
                'confidence':conf})
        hi=sum(1 for r in rows if r['confidence']=='HIGH'); md=sum(1 for r in rows if r['confidence']=='MEDIUM')
        sys.stderr.write(f'[m2] {g}: {len(rows)} -> HIGH={hi} MED={md} UNM={len(rows)-hi-md}\n')
        out[g]=rows
    json.dump(out, open(a.out,'w'), indent=2)
    sys.stderr.write(f'[m2] wrote {a.out}\n')

if __name__=='__main__':
    main()
