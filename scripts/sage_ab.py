#!/usr/bin/env python3
"""A/B summary for the SAGE EVPI-gate experiment (user-mode telecom).

Compares two run dirs (SAGE off vs on) on:
  * pass rate (reward==1)
  * avg # user-side instructions per task  (the paper's #Q efficiency metric)
  * avg # agent tool calls per task
  * give-up breakdown (evpi_below_threshold / max_asks / stuck-3)
Reads results.json (reward) + state_log.jsonl (exec/sage entries).

Usage: python scripts/sage_ab.py data/simulations/sage_off_8b data/simulations/sage_on_8b
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_rewards(run: Path):
    """task_id -> (reward, agent_cost)."""
    rj = run / "results.json"
    if not rj.exists():
        return {}
    data = json.loads(rj.read_text())
    sims = data.get("simulations", data) if isinstance(data, dict) else data
    out = {}
    for s in sims:
        tid = s.get("task_id") or s.get("id")
        rw = s.get("reward")
        if rw is None:
            rw = (s.get("reward_info") or {}).get("reward")
        out[tid] = (rw, s.get("agent_cost") or 0.0)
    return out


def summarize(run: Path):
    rewards = load_rewards(run)
    rows = [json.loads(l) for l in (run / "state_log.jsonl").read_text().splitlines() if l.strip()]
    instr = Counter()       # task -> #user-side instructions
    toolcalls = Counter()   # task -> #agent tool calls
    giveups = Counter()     # reason -> count
    tasks = set()
    for r in rows:
        tid = r.get("task_id")
        tasks.add(tid)
        if r["kind"] == "exec":
            if r.get("side") == "user":
                instr[tid] += 1
            elif r.get("side") == "agent":
                toolcalls[tid] += 1
        elif r["kind"] == "sage_giveup":
            giveups[r.get("reason", "?")] += 1
        elif r["kind"] == "stuck_skip":
            giveups["stuck_skip"] += 1
    n = max(len(rewards) or len(tasks), 1)
    passed = sum(1 for (rw, _) in rewards.values() if rw == 1)
    agent_cost = sum(c for (_, c) in rewards.values())
    return {
        "n_tasks": len(rewards) or len(tasks),
        "pass": passed,
        "pass_rate": passed / n,
        "avg_instr": sum(instr.values()) / n,
        "avg_toolcalls": sum(toolcalls.values()) / n,
        "avg_agent_cost": agent_cost / n,
        "giveups": dict(giveups),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    off, on = Path(sys.argv[1]), Path(sys.argv[2])
    so, sn = summarize(off), summarize(on)
    print(f"{'metric':<22}{'OFF (magic-3)':>18}{'ON (EVPI gate)':>18}")
    print("-" * 58)
    for k, label in [("pass_rate", "pass rate"), ("pass", "passed"), ("n_tasks", "n tasks"),
                     ("avg_instr", "avg #instructions"), ("avg_toolcalls", "avg #tool calls"),
                     ("avg_agent_cost", "avg agent $")]:
        fa = f"{so[k]:.3f}" if isinstance(so[k], float) else str(so[k])
        fb = f"{sn[k]:.3f}" if isinstance(sn[k], float) else str(sn[k])
        print(f"{label:<22}{fa:>18}{fb:>18}")
    print(f"\ngive-ups OFF: {so['giveups']}")
    print(f"give-ups ON : {sn['giveups']}")
    di = sn["avg_instr"] - so["avg_instr"]
    dp = sn["pass_rate"] - so["pass_rate"]
    print(f"\nΔ pass_rate = {dp:+.3f}   Δ avg_instr = {di:+.2f} "
          f"({'fewer' if di < 0 else 'more'} questions/task)")


if __name__ == "__main__":
    main()
