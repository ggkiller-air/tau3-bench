#!/usr/bin/env python
"""Multi-trial pass^k analysis for retail standard-mode runs.

Usage: python analyze_retail_trials.py <sim_dir> [<sim_dir> ...]
Each dir's results.json may contain N trials per task (--num-trials N). Reports per-config:
  - overall pass^1 (mean over all sims) and per-trial pass^1 (mean ± spread = noise estimate)
  - pass^k for k=1..N (C(success,k)/C(N,k) averaged over tasks with N trials)
  - termination breakdown; reward=0-but-db_match=True count (judge/nl_assertion or extra-action)
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb

def load(d):
    p = d if d.endswith(".json") else os.path.join("data/simulations", d, "results.json")
    if not os.path.exists(p):
        p = os.path.join(d, "results.json")
    raw = json.load(open(p))
    return raw["simulations"] if isinstance(raw, dict) and "simulations" in raw else raw

def analyze(d):
    sims = load(d)
    by_task = defaultdict(list)     # task_id -> [reward(0/1) per trial]
    by_trial = defaultdict(list)    # trial -> [reward per sim]
    term = Counter()
    db_true_rew0 = 0
    n = 0
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        ok = 1 if r == 1.0 else 0
        tid = str(s.get("task_id")); tr = s.get("trial")
        by_task[tid].append(ok)
        by_trial[tr].append(ok)
        term[s.get("termination_reason")] += 1
        if ok == 0 and (ri.get("db_check") or {}).get("db_match") is True:
            db_true_rew0 += 1
        n += 1
    # per-trial pass^1
    trial_p1 = {tr: sum(v) / len(v) for tr, v in sorted(by_trial.items())}
    overall_p1 = sum(sum(v) for v in by_task.values()) / max(n, 1)
    # pass^k over tasks that have >= k trials
    Ntr = max((len(v) for v in by_task.values()), default=1)
    passk = {}
    for k in range(1, Ntr + 1):
        num = den = 0.0
        for v in by_task.values():
            N = len(v)
            if N < k:
                continue
            c = sum(v)
            num += comb(c, k) / comb(N, k) if c >= k else 0.0
            den += 1
        passk[k] = num / den if den else 0.0
    print(f"\n==== {d} ====")
    print(f"  sims={n}  tasks={len(by_task)}  trials/task≈{Ntr}")
    print(f"  overall pass^1 = {overall_p1:.3f}")
    print(f"  per-trial pass^1 = " + ", ".join(f"t{tr}:{p:.3f}" for tr, p in trial_p1.items())
          + (f"  (spread {max(trial_p1.values())-min(trial_p1.values()):.3f} = noise)" if len(trial_p1) > 1 else ""))
    print(f"  pass^k = " + ", ".join(f"^{k}:{passk[k]:.3f}" for k in passk))
    print(f"  termination: {dict(term)}")
    print(f"  reward=0 但 db_match=True (judge/nl/extra-action 噪声): {db_true_rew0}")
    return overall_p1, passk

if __name__ == "__main__":
    for d in sys.argv[1:]:
        analyze(d)
