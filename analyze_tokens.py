#!/usr/bin/env python3
"""pass^4 衰减 token/上下文分析。

用法: python analyze_tokens.py <sim_dir>
  <sim_dir>/results.json  + tokens.json (SCHEMAFLEX_TOKEN_LOG) [+ token_trace.jsonl]

按 (task_id, seed) 把 per-sim token/上下文 join 到 reward，回答：
  ① 失败 trial 的上下文(peak_prompt_exec)/回合数/总 token 是否系统性高于成功 trial
  ② 同一 task 内 pass/fail 混合(=pass^k 衰减源)的任务，逐 trial 对比
"""
import json, sys, os
from collections import defaultdict

def load_sims(d):
    r = json.load(open(os.path.join(d, "results.json")))
    return r["simulations"] if isinstance(r, dict) and "simulations" in r else r

def main(d):
    sims = load_sims(d)
    tok = {}
    tpath = os.path.join(d, "tokens.json")
    if os.path.exists(tpath):
        tok = json.load(open(tpath))
    # 索引 token 记录: (task_id, seed) -> rec
    by_key = {}
    for k, rec in tok.items():
        by_key[(rec.get("task_id"), rec.get("seed"))] = rec

    rows = []  # (task_id, seed, trial, reward, calls, peak_exec, peak_any, total)
    for s in sims:
        tid = s.get("task_id"); seed = s.get("seed"); trial = s.get("trial")
        rw = (s.get("reward_info", {}) or {}).get("reward"); rw = 0.0 if rw is None else rw
        rec = by_key.get((tid, seed)) or {}
        total = (rec.get("prompt", 0) + rec.get("completion", 0))
        rows.append({"tid": tid, "seed": seed, "trial": trial, "rw": rw,
                     "calls": rec.get("calls", 0), "peak_exec": rec.get("peak_prompt_exec", 0),
                     "peak_any": rec.get("peak_prompt", 0), "total": total,
                     "term": s.get("termination_reason") or s.get("termination")})

    def stat(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs)/len(xs)) if xs else 0

    P = [r for r in rows if r["rw"] == 1.0]
    F = [r for r in rows if r["rw"] != 1.0]
    print(f"=== 全体 {len(rows)} sims: pass {len(P)} / fail {len(F)} ===")
    for label, grp in (("PASS", P), ("FAIL", F)):
        print(f"  {label}: calls均={stat([r['calls'] for r in grp]):.1f}  "
              f"peak_exec均={stat([r['peak_exec'] for r in grp]):.0f}  "
              f"peak_any均={stat([r['peak_any'] for r in grp]):.0f}  "
              f"total均={stat([r['total'] for r in grp]):.0f}")

    # 按 task 分组找 pass^k 衰减源(pass/fail 混合)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["tid"]].append(r)
    mixed = {t: v for t, v in by_task.items() if 0 < sum(1 for r in v if r["rw"] == 1.0) < len(v)}
    print(f"\n=== pass^k 衰减源: {len(mixed)} 个 task 在 trial 间 pass/fail 混合 ===")
    for t, v in sorted(mixed.items()):
        short = t.split("]")[-1][:34]
        pv = [r for r in v if r["rw"] == 1.0]; fv = [r for r in v if r["rw"] != 1.0]
        print(f"  {short:36s} pass {len(pv)}/{len(v)} | "
              f"PASS peak_exec={stat([r['peak_exec'] for r in pv]):.0f} calls={stat([r['calls'] for r in pv]):.1f}"
              f"  vs  FAIL peak_exec={stat([r['peak_exec'] for r in fv]):.0f} calls={stat([r['calls'] for r in fv]):.1f}"
              f"  FAIL_term={[r['term'] for r in fv]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python analyze_tokens.py <sim_dir>"); sys.exit(1)
    main(sys.argv[1])
