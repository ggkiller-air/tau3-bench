#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_arm.py — 解析单个 arm 的产物，打印 pass^k / 家族 / 终止原因 / A1-A2-B 分桶 /
state_log 事件分布 / supervisor 决策分解 / token 桶。可选第二个 arm 做 delta 对照。

用法:
  python analyze_arm.py <save_dir>                 # 例: data/simulations/user_replan_sup_full
  python analyze_arm.py <save_dir> <baseline_dir>  # 与对照 arm 比 pass^k delta
仅读取产物文件，不跑任何评测。
"""
import sys, os, json, re, math, collections

_MUT_RE = re.compile(
    r"^(toggle_|enable_|disable_|set_|make_|reset_|reseat_|unseat_|lock_|unlock_|"
    r"refuel_|grant_|revoke_|disconnect_|resume_|reboot_|send_payment|pay_)"
)
_RO_RE = re.compile(r"^(get_|check_|run_speed_test|can_send_mms)")
_FAM_RE = re.compile(r"^\[([^\]]+)\]")
_PERSONA_RE = re.compile(r"\[PERSONA:([^\]]+)\]")
SUCCESS = 0.999


def _family(task_id):
    m = _FAM_RE.match(task_id or "")
    return m.group(1) if m else "?"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _passk(succ_by_task, k):
    """exact combinatorial pass^k averaged over tasks with >=k trials."""
    vals = []
    for tid, (c, n) in succ_by_task.items():
        if n < k:
            continue
        vals.append(math.comb(c, k) / math.comb(n, k))
    return (sum(vals) / len(vals)) if vals else float("nan")


def _scan_sim_tools(sim):
    """returns (has_mutating, n_mut, n_readonly) by scanning the sim's own messages."""
    n_mut = n_ro = 0
    for m in sim.get("messages") or []:
        if not isinstance(m, dict):
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") if isinstance(tc, dict) else None
            if not nm:
                continue
            if _MUT_RE.match(nm):
                n_mut += 1
            elif _RO_RE.match(nm):
                n_ro += 1
    return (n_mut > 0, n_mut, n_ro)


def analyze(save_dir):
    res = _load(os.path.join(save_dir, "results.json"))
    sims = res["simulations"]
    out = {"dir": save_dir, "n_sims": len(sims)}

    # --- pass^k (overall + per family) ---
    succ = collections.defaultdict(lambda: [0, 0])          # task_id -> [success, trials]
    succ_fam = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    term = collections.Counter()
    buckets = collections.Counter()
    fam_fail = collections.defaultdict(collections.Counter)
    for s in sims:
        tid = s["task_id"]
        fam = _family(tid)
        r = (s.get("reward_info") or {}).get("reward", 0.0) or 0.0
        ok = r >= SUCCESS
        succ[tid][1] += 1
        succ[tid][0] += int(ok)
        succ_fam[fam][tid][1] += 1
        succ_fam[fam][tid][0] += int(ok)
        tr = s.get("termination_reason", "?")
        term[tr] += 1
        if not ok:
            has_mut, _, _ = _scan_sim_tools(s)
            if tr == "max_steps":
                b = "A1_repair_no_close" if has_mut else "A2_never_repaired"
            elif tr == "user_stop":
                b = "B_user_stop"
            else:
                b = "other_fail"
            buckets[b] += 1
            fam_fail[fam][b] += 1

    succ = {k: tuple(v) for k, v in succ.items()}
    out["passk"] = {f"pass^{k}": round(_passk(succ, k), 4) for k in (1, 2, 3, 4)}
    out["per_family_pass1"] = {}
    for fam, d in sorted(succ_fam.items()):
        d2 = {k: tuple(v) for k, v in d.items()}
        ntri = sum(n for _, n in d2.values())
        nok = sum(c for c, _ in d2.values())
        out["per_family_pass1"][fam] = {"pass^1": round(_passk(d2, 1), 4),
                                         "ok/trials": f"{nok}/{ntri}"}
    out["termination"] = dict(term)
    out["fail_buckets"] = dict(buckets)
    out["fail_buckets_by_family"] = {f: dict(c) for f, c in fam_fail.items()}

    # --- state_log events + supervisor decision breakdown ---
    slp = os.path.join(save_dir, "state_log.jsonl")
    ev = collections.Counter()
    sup = collections.Counter()           # decision -> count
    sup_valid = [0, 0]                     # valid_target true/total for schedule
    sup_written = 0                        # total fields written by extract
    sup_calls = 0
    if os.path.exists(slp):
        for line in open(slp):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k = e.get("kind", "?")
            ev[k] += 1
            if k == "supervise":
                sup_calls += 1
                dec = e.get("decision", "?")
                sup[dec] += 1
                if dec == "schedule":
                    sup_valid[1] += 1
                    sup_valid[0] += int(bool(e.get("valid_target")))
                if dec == "extract":
                    w = e.get("written")
                    sup_written += (len(w) if isinstance(w, (list, dict)) else (w or 0)) if w else 0
    out["state_events"] = dict(ev)
    out["supervisor"] = {
        "total_calls": sup_calls,
        "decisions": dict(sup),
        "schedule_valid_target": f"{sup_valid[0]}/{sup_valid[1]}",
        "extract_fields_written": sup_written,
        "reschedule_events": ev.get("reschedule", 0),
        "escalate_events": ev.get("escalate", 0),
    }

    # --- tokens (defensive: sum whatever buckets exist) ---
    tkp = os.path.join(save_dir, "tokens.json")
    if os.path.exists(tkp):
        tk = _load(tkp)
        agg = collections.Counter()
        for _tid, rec in tk.items():
            if isinstance(rec, dict):
                for kk, vv in rec.items():
                    if isinstance(vv, (int, float)):
                        agg[kk] += vv
        out["tokens"] = dict(agg)
        out["tokens_note"] = ("supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor"
                              "（当前不记账）；supervisor 调用次数以 state_events.supervise 为准")
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    arm = analyze(sys.argv[1])
    print(json.dumps(arm, ensure_ascii=False, indent=2))
    if len(sys.argv) >= 3:
        base = analyze(sys.argv[2])
        print("\n=== DELTA vs %s ===" % sys.argv[2])
        for k in ("pass^1", "pass^2", "pass^3", "pass^4"):
            a, b = arm["passk"][k], base["passk"][k]
            print(f"  {k}: {b:.4f} -> {a:.4f}  ({a-b:+.4f})")
        print("  termination:", dict(base["termination"]), "->", dict(arm["termination"]))
        print("  fail_buckets:", dict(base["fail_buckets"]), "->", dict(arm["fail_buckets"]))


if __name__ == "__main__":
    main()
