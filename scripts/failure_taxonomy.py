"""Failure-stage / outcome annotation for telecom strat90 runs (Figure 5 data).

Usage: python scripts/failure_taxonomy.py <run_dir> <out_csv>
e.g.   python scripts/failure_taxonomy.py data/simulations/strat90_hilo reports/ours_strat90_failure.csv

Method-agnostic: works off results.json messages only (state_log.jsonl, if present,
adds ours-only evidence columns). All labels are mechanical — no judgment calls:

Funnel stages are CUMULATIVE REACH: reaching stage k implies passing 1..k-1,
enforced by construction (s1 = s2 OR evidence1, s2 = s3 OR evidence2, ...), so
the S4 bar equals the pass rate exactly and the funnel is monotone.

  S1 diagnose  : every gold fault atom (parsed from task_id) was either probed
                 (mapped diagnostic tool called) or acted on (its mapped repair
                 tool called) -- repairing a fault implies having identified it.
  S2 repair    : every WRITE action in evaluation_criteria.actions was matched
                 in the trace (name equal, gold args a subset of call args).
                 For unfixable faults the gold write IS transfer_to_human_agents.
  S3 verify    : the LAST call of the family's verification tool returned the
                 goal state (mobile_data: speed-test Excellent; service:
                 connection 'connected'; mms: 'can send MMS'). If the gold plan
                 is a transfer (unfixable fault), executing the transfer counts.
  S4 success   : reward == 1.0.

Outcome (primary) = success | fail@diagnose | fail@repair | fail@verify | fail@final.
Mechanism (secondary) = infra | loop | premature_closure | exhausted | none.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

TASKS = "data/tau2/domains/telecom/tasks_strat90.json"

# fault atom -> any-of diagnostic tools (weak entries flagged in comments)
DIAG = {
    "airplane_mode_on": {"check_status_bar", "check_network_status"},
    "bad_network_preference": {"check_network_mode_preference"},
    "bad_vpn": {"check_vpn_status", "check_status_bar"},
    "bad_wifi_calling": {"check_wifi_calling_status"},
    "break_apn_settings": {"check_apn_settings"},
    "break_apn_mms_setting": {"check_apn_settings"},
    "break_app_sms_permission": {"check_app_permissions"},
    "break_app_storage_permission": {"check_app_permissions"},
    "break_app_both_permissions": {"check_app_permissions"},
    "data_mode_off": {"check_status_bar", "check_network_status"},
    "data_saver_mode_on": {"check_data_restriction_status", "check_status_bar"},
    "data_usage_exceeded": {"get_data_usage"},
    "lock_sim_card_pin": {"check_sim_status"},
    "unseat_sim_card": {"check_sim_status"},
    "overdue_bill_suspension": {"get_bills_for_customer", "check_payment_request"},
    # weak: contract status only visible via customer/line details lookups
    "contract_end_suspension": {"get_details_by_id", "get_customer_by_phone",
                                "get_customer_by_id", "get_customer_by_name"},
    "user_abroad_roaming_disabled_off": {"check_network_status", "check_status_bar"},
    "user_abroad_roaming_disabled_on": {"check_network_status", "check_status_bar"},
    "user_abroad_roaming_enabled_off": {"check_network_status", "check_status_bar"},
}

# fault atom -> repair tools; calling one implies the fault was identified (S1)
REPAIR = {
    "airplane_mode_on": {"turn_airplane_mode_off", "toggle_airplane_mode"},
    "bad_network_preference": {"set_network_mode_preference"},
    "bad_vpn": {"disconnect_vpn"},
    "bad_wifi_calling": {"set_wifi_calling", "toggle_wifi_calling"},
    "break_apn_settings": {"reset_apn_settings", "set_apn_settings"},
    "break_apn_mms_setting": {"reset_apn_settings", "set_apn_settings"},
    "break_app_sms_permission": {"grant_app_permission"},
    "break_app_storage_permission": {"grant_app_permission"},
    "break_app_both_permissions": {"grant_app_permission"},
    "data_mode_off": {"turn_data_on", "toggle_data"},
    "data_saver_mode_on": {"turn_data_saver_mode_off", "toggle_data_saver_mode"},
    "data_usage_exceeded": {"refuel_data"},
    "lock_sim_card_pin": {"transfer_to_human_agents"},      # unfixable
    "unseat_sim_card": {"reseat_sim_card"},
    "overdue_bill_suspension": {"send_payment_request", "make_payment", "resume_line"},
    "contract_end_suspension": {"transfer_to_human_agents"},  # unfixable
    "user_abroad_roaming_disabled_off": {"enable_roaming", "toggle_roaming", "turn_roaming_on"},
    "user_abroad_roaming_disabled_on": {"enable_roaming", "toggle_roaming", "turn_roaming_on"},
    "user_abroad_roaming_enabled_off": {"turn_roaming_on", "toggle_roaming"},
}

READ_PREFIXES = ("check_", "get_", "run_", "can_", "assert_", "simulate_")

# family -> (verify tool, goal regex on its result)
VERIFY = {
    "mobile_data_issue": ("run_speed_test", re.compile(r"\(Excellent\)")),
    "service_issue": ("check_network_status", re.compile(r"Cellular Connection:\s*connected", re.I)),
    "mms_issue": ("can_send_mms", re.compile(r"app can send MMS", re.I)),
}

LOOP_REPEAT_MIN = 5  # identical tool name called >= this many times, under max_steps


def parse_task_id(tid: str):
    body = re.sub(r"\[PERSONA:[^\]]*\]$", "", tid)
    m = re.match(r"\[(\w+)\](.*)", body)
    fam, faults = m.group(1), m.group(2).split("|")
    persona = re.search(r"\[PERSONA:([^\]]*)\]", tid)
    return fam, faults, (persona.group(1) if persona else "")


def iter_calls(messages):
    """(index, tool_name, args) for every tool call by any participant."""
    for i, m in enumerate(messages):
        for tc in (m.get("tool_calls") or []):
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            yield i, tc["name"], args


def tool_results(messages, name):
    """Result strings of tool messages responding to calls of `name`, in order."""
    pending = {}  # tool_call id -> name
    out = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = tc["name"]
        if m.get("role") == "tool" and pending.get(m.get("tool_id") or m.get("id")) == name:
            out.append(str(m.get("content")))
    if out:
        return out
    # fallback: positional pairing (tool message follows its call)
    calls_at = [i for i, n, _ in iter_calls(messages) if n == name]
    for i in calls_at:
        for j in range(i + 1, min(i + 4, len(messages))):
            if messages[j].get("role") == "tool":
                out.append(str(messages[j].get("content")))
                break
    return out


def args_match(gold: dict, actual: dict) -> bool:
    for k, v in (gold or {}).items():
        if str(actual.get(k)) != str(v):
            return False
    return True


def classify(sim, task):
    tid = sim["task_id"]
    fam, faults, persona = parse_task_id(tid)
    msgs = sim.get("messages") or []
    reward = (sim.get("reward_info") or {}).get("reward")
    term = sim.get("termination_reason")
    calls = list(iter_calls(msgs))
    call_names = [n for _, n, _ in calls]
    name_counts = Counter(call_names)

    names = set(call_names)

    # per-stage evidence
    diag_missing = [f for f in faults if not ((DIAG[f] | REPAIR[f]) & names)]
    evid1 = not diag_missing

    gold_writes = [a for a in (task["evaluation_criteria"].get("actions") or [])
                   if not a["name"].startswith(READ_PREFIXES)]
    missing_writes = []
    for a in gold_writes:
        hit = any(n == a["name"] and args_match(a.get("arguments"), ar)
                  for _, n, ar in calls)
        if not hit:
            missing_writes.append(a["name"])
    evid2 = not missing_writes

    gold_is_transfer = any(a["name"] == "transfer_to_human_agents" for a in gold_writes)
    vtool, goal_re = VERIFY[fam]
    vres = tool_results(msgs, vtool)
    if gold_is_transfer:
        evid3 = "transfer_to_human_agents" in names
    else:
        evid3 = bool(vres) and bool(goal_re.search(vres[-1]))

    # cumulative reach: later stage implies earlier ones
    s4 = reward == 1.0
    s3 = s4 or evid3
    s2 = s3 or evid2
    s1 = s2 or evid1

    stage = 0
    for ok in (s1, s2, s3, s4):
        if not ok:
            break
        stage += 1

    outcome = ["fail@diagnose", "fail@repair", "fail@verify", "fail@final", "success"][stage]

    if term == "infrastructure_error":
        mech = "infra"
    elif term == "too_many_errors":
        mech = "invalid_tool_calls"   # e.g. agent calls user-side tools it doesn't own
    elif term == "max_steps":
        mech = "loop" if name_counts and name_counts.most_common(1)[0][1] >= LOOP_REPEAT_MIN else "exhausted"
    elif not s4 and not s2:
        mech = "premature_closure"   # ended normally with gold repairs missing
    elif not s4:
        mech = "premature_closure"   # ended normally, repairs done but goal unverified/unmet
    else:
        mech = "none"

    top_tool, top_n = (name_counts.most_common(1)[0] if name_counts else ("", 0))
    return {
        "task_id": tid, "family": fam, "persona": persona,
        "n_faults": len(faults), "faults": "|".join(faults),
        "reward": reward, "termination_reason": term,
        "n_messages": len(msgs), "n_tool_calls": len(calls),
        "s1_diagnose": int(s1), "s2_repair": int(s2),
        "s3_verify": int(s3), "s4_success": int(s4),
        "stage_reached": stage, "outcome": outcome, "mechanism": mech,
        "diag_missing": "|".join(diag_missing),
        "gold_writes_total": len(gold_writes),
        "gold_writes_missing": "|".join(missing_writes),
        "last_verify_result": (vres[-1][:120].replace("\n", " ") if vres else ""),
        "top_repeated_tool": top_tool, "top_repeat_count": top_n,
    }


def main(run_dir, out_csv):
    tasks = {t["id"]: t for t in json.load(open(TASKS))}
    res = json.load(open(f"{run_dir}/results.json"))
    sims = res["simulations"] if isinstance(res, dict) else res

    state_extra = {}
    slog = Path(run_dir) / "state_log.jsonl"
    if slog.exists():
        agg = {}
        for line in open(slog):
            e = json.loads(line)
            d = agg.setdefault(e.get("task_id"), Counter())
            d[e.get("kind")] += 1
        state_extra = {tid: {"n_replan": c.get("replan", 0), "n_hilo": c.get("hilo", 0),
                             "n_close": c.get("close", 0), "n_stuck_skip": c.get("stuck_skip", 0)}
                       for tid, c in agg.items()}

    rows = []
    for s in sims:
        if s["task_id"] not in tasks:
            continue
        row = classify(s, tasks[s["task_id"]])
        row.update(state_extra.get(s["task_id"], {"n_replan": "", "n_hilo": "", "n_close": "", "n_stuck_skip": ""}))
        rows.append(row)
    rows.sort(key=lambda r: (r["family"], r["n_faults"], r["task_id"]))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"{run_dir}: {n} episodes -> {out_csv}")
    for k in ("s1_diagnose", "s2_repair", "s3_verify", "s4_success"):
        print(f"  cumulative {k}: {sum(r[k] for r in rows)}/{n}")
    reach = Counter(r["stage_reached"] for r in rows)
    cum = [sum(v for s, v in reach.items() if s >= k) for k in (1, 2, 3, 4)]
    print(f"  funnel (monotone reach S1..S4): {cum}")
    print("  outcome:", dict(Counter(r["outcome"] for r in rows)))
    print("  mechanism:", dict(Counter(r["mechanism"] for r in rows)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
