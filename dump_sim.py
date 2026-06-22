#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dump_sim.py <arm> <sim_id>  —— 打印某个 sim 的任务定义/期望/agent 可见全轨迹。
仅读取 data/simulations/<arm>/results.json，不跑任何评测。
例: python dump_sim.py user_replan_sup_full 21080ca2-9b9d-48bf-9fd7-dfded7f6bd9b
"""
import sys, json, os

def main():
    arm, sid = sys.argv[1], sys.argv[2]
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "simulations", arm)
    d = json.load(open(os.path.join(base, "results.json")))
    sims = d["simulations"]
    s = next((x for x in sims if x["id"] == sid), None)
    if s is None:
        print("NOT FOUND", sid); sys.exit(1)
    tid = s["task_id"]
    print("=" * 80)
    print("ARM:", arm, "| SIM:", sid, "| trial:", s.get("trial"))
    print("TASK_ID:", tid)
    print("TERMINATION:", s.get("termination_reason"), "| reward:", (s.get("reward_info") or {}).get("reward"))

    # task definition (purpose / ticket / scenario) if present
    task = next((t for t in d.get("tasks", []) if t.get("id") == tid), None)
    if task:
        print("\n--- TASK DEFINITION (ground truth) ---")
        for k in ("purpose", "description", "scenario", "ticket", "user_scenario", "initial_state", "evaluation_criteria"):
            if k in task and task[k]:
                print(f"[{k}] {json.dumps(task[k], ensure_ascii=False)[:1200]}")

    # expected end-state / action checks from reward_info
    ri = s.get("reward_info") or {}
    print("\n--- EXPECTED (reward_info) ---")
    for k in ("db_check", "action_checks", "env_assertions"):
        v = ri.get(k)
        if v:
            print(f"[{k}] {json.dumps(v, ensure_ascii=False)[:1400]}")
    rb = ri.get("reward_breakdown")
    if rb:
        print("[reward_breakdown]", json.dumps(rb, ensure_ascii=False)[:600])

    # full agent-visible transcript
    print("\n--- TRANSCRIPT (role/requestor : content | tool_calls) ---")
    for i, m in enumerate(s.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role"); req = m.get("requestor")
        content = (m.get("content") or "")
        if isinstance(content, str):
            content = content.strip().replace("\n", " ")
        tcs = m.get("tool_calls") or []
        head = f"[{i:02d}] {role}/{req}"
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    print(f"{head}  CALL {tc.get('name')}({json.dumps(tc.get('arguments'), ensure_ascii=False)})")
        if content:
            err = " [ERROR]" if m.get("error") else ""
            print(f"{head}{err}  {content[:500]}")

if __name__ == "__main__":
    main()
