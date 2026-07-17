#!/usr/bin/env python
"""Per-task pass^1 / termination breakdown for retail runs, optional slice filter.

Usage:
  python analyze_retail.py <sim_dir> [<sim_dir> ...] [--slice "24 25 0 1"]
Compares multiple run dirs task-by-task on the (optionally sliced) task set.
"""
import json, os, sys
from collections import Counter

def load(p):
    d = json.load(open(p))
    sims = d["simulations"] if isinstance(d, dict) and "simulations" in d else d
    out = {}
    for s in sims:
        tid = str(s.get("task_id"))
        ri = s.get("reward_info") or {}
        out[tid] = (ri.get("reward"), s.get("termination_reason"))
    return out

def main():
    args = sys.argv[1:]
    sl = None
    if "--slice" in args:
        i = args.index("--slice"); sl = set(args[i+1].split()); args = args[:i] + args[i+2:]
    dirs = args
    runs = {}
    for d in dirs:
        p = os.path.join("data/simulations", d, "results.json") if not d.endswith(".json") else d
        if not os.path.exists(p):
            p = os.path.join(d, "results.json")
        runs[d] = load(p)
    keys = set()
    for r in runs.values(): keys |= set(r)
    if sl: keys &= sl
    keys = sorted(keys, key=lambda x: (len(x), x))
    print(f"{'task':>6} | " + " | ".join(f"{d[:18]:>18}" for d in dirs))
    for t in keys:
        cells = []
        for d in dirs:
            v = runs[d].get(t)
            if v is None: cells.append(f"{'--':>18}")
            else:
                r, term = v
                mark = "✓" if r == 1.0 else ("·" if term == "max_steps" else "✗")
                cells.append(f"{mark} {str(r):>5} {term[:9]:>9}")
        print(f"{t:>6} | " + " | ".join(cells))
    print("-"*60)
    for d in dirs:
        sub = {t: runs[d][t] for t in keys if t in runs[d]}
        succ = sum(1 for v in sub.values() if v[0] == 1.0)
        term = Counter(v[1] for v in sub.values() if v[0] != 1.0)
        print(f"{d}: pass^1 = {succ}/{len(sub)} = {succ/max(len(sub),1):.3f}  | fails: {dict(term)}")

if __name__ == "__main__":
    main()
